import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import precision_recall_fscore_support



# from custom_layers import Encoder, Decoder, Extracter, Estimator



def random_batch(X, y=None, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx]

def progress_bar(iteration, total, size=30):
    running = iteration < total
    c = ">" if running else "="
    p = (size - 1) * iteration // total
    fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
    params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
    return fmt.format(*params)

def print_status_bar(iteration, total, loss, metrics=None, size=30):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{} - {}".format(progress_bar(iteration, total), metrics), end=end)
    
def euclid_norm(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))


class Encoder(keras.layers.Layer):
    # encoder layers of compression network
    def __init__(self, hidden_layer_sizes,activation="elu", kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
        self.hidden= [keras.layers.Dense(size, activation=activation, kernel_initializer=kernel_initializer,
                                         kernel_regularizer=keras.regularizers.l2(0.01)) for size in hidden_layer_sizes[:-1]]
        self.out = keras.layers.Dense(hidden_layer_sizes[-1], activation=activation,
                                      kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, x):
        z = keras.layers.Flatten()(x)
        for layer in self.hidden:
            z = layer(z)
        z = self.out(z)
        return z

class Decoder(keras.layers.Layer):
    # decoder layers of compresion network
    def __init__(self, hidden_layer_sizes, input_size, activation="elu", kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(size, activation=activation, kernel_initializer=kernel_initializer, 
                                          kernel_regularizer=keras.regularizers.l2(0.01)) 
                       for size in hidden_layer_sizes[:-1][::-1]]
        self.out = keras.layers.Dense(input_size, activation=keras.activations.linear,
                                      kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, z):
        for layer in self.hidden:
            z = layer(z)
        x_res = self.out(z)
        return x_res


class Extracter(keras.layers.Layer):
    # Defining the Custom Error Construction Layer between the Input Layer and Output Layer of the Autoencoder
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, x_res, z_c = inputs
        # Calculate Euclid norm, distance
        norm_x = euclid_norm(x)
        norm_x_dash = euclid_norm(x_res)
        dist_x = euclid_norm(x - x_res)
        dot_x = tf.reduce_sum(x - x_res, axis=1)

        #  1. loss_E : relative Euclidean distance
        #  2. loss_C : cosine similarity
        min_val = 1e-3
        loss_E = dist_x / (norm_x + min_val)
        loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x_dash + min_val))
        return tf.concat([loss_E[:,np.newaxis], loss_C[:,np.newaxis], z_c], axis=1)

class Estimator(keras.layers.Layer):
    # estimation network
    def __init__(self, hidden_layer_sizes, activation="elu", kernel_initializer="he_normal", dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.dropout_layer = keras.layers.Dropout(rate=dropout_rate)
        self.hidden = [keras.layers.Dense(size, activation=activation, kernel_initializer=kernel_initializer,
                                          kernel_regularizer=keras.regularizers.l2(0.01))
                               for size in hidden_layer_sizes[:-1]]
        self.out = keras.layers.Dense(hidden_layer_sizes[-1], activation=keras.activations.softmax,
                                      kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, z):
        for layer in self.hidden:
            z = layer(z)
            z = self.dropout_layer(z)
        output = self.out(z)
        return output
    

class DAMPPCA:
    """ Deep Autoencoding Mixture of Probabilistic Principal Component Analyzers.
    """

    MODEL_FILENAME = "DAMPPCA_model"
    SCALER_FILENAME = "DAMPPCA_scaler"

    def __init__(self, input_size, comp_hiddens, est_hiddens, comp_activation,
                 est_activation, kernel_initializer, est_dropout_ratio=0.2,n_epochs=1000, batch_size = 128,
                 lambda1=1, lambda2=0.1, lambda3=1, lambda4=0.0001, learning_rate=0.01, 
                 patience=10, normalize='MinMax', random_seed=42):
        
        inputs = keras.layers.Input(input_size, name="input")
        self.encoder = Encoder(comp_hiddens, comp_activation, kernel_initializer)
        self.decoder = Decoder(comp_hiddens, input_size, comp_activation, kernel_initializer)
        codings = self.encoder(inputs)
        recons = self.decoder(codings)
        self.comp_network = keras.models.Model(inputs=[inputs], outputs=[recons])
        self.extracter = Extracter()
        latents = self.extracter((inputs, recons, codings))
        self.est_network = Estimator(est_hiddens, est_activation, kernel_initializer, est_dropout_ratio)
        gamma = self.est_network(latents)
        self.damppca= keras.models.Model(inputs=[inputs], outputs=[gamma])

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.W = None
        self.sigma2 = None
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        
        self.lr = learning_rate
        self.patience = patience
        
        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed
        
        self.phi_list = []
        self.mu_list = []
        self.L_list = []


    def custom_loss(self, inputs):
        # calculate the loss function with three loss components
        recons = self.comp_network(inputs)  
        
        # (1) reconstruction loss of DAE 
        dae_loss = self.lambda1 *tf.reduce_mean(tf.reduce_sum(tf.square(inputs - recons), axis=1), axis=0)
        
        # (2) reconstruction probability loss
        res_probs, A, C, latent_error = self.res_prob(inputs)
        res_prob_loss = self.lambda2 * tf.reduce_mean(res_probs)
        
        # (3) reconstruction loss of MMPCA 
        pca_loss = self.lambda3 * tf.reduce_mean(latent_error, axis=0)
        
        # (4) sigularity penalty
        diag_loss = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(A))) + tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(C)))
        cov_loss = self.lambda4 * diag_loss
        
        # sum of three loss components
        loss = dae_loss + res_prob_loss + pca_loss + cov_loss
        
#         print("\n dae_loss: {} \n res_prob_loss: {}, \n pca_loss: {}" .format(dae_loss, res_prob_loss, pca_loss))

        return loss
    
    def para_init(self, inputs): 
        
        if self.normalize=='MinMax':
            self.scaler = MinMaxScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
            
        elif self.normalize=='Standard':
            self.scaler = StandardScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
            
        codings = self.encoder(inputs)
        recons = self.decoder(codings)
        z = self.extracter((inputs, recons, codings))
        d = z.shape[-1]
        gamma = self.est_network(z)
        
        # Calculate mu, sigma
        # i   : index of samples
        # k   : index of components
        # l,m : index of features
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        phi = tf.reduce_mean(gamma, axis=0)
        mu =  tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:, np.newaxis]
        z_centered_1 = tf.sqrt(gamma[:, :, np.newaxis]) * (z[:, np.newaxis, :] - mu[np.newaxis, :, :])
        S =  tf.einsum('ikl,ikm->klm', z_centered_1, z_centered_1) / gamma_sum[:, np.newaxis, np.newaxis] 
        q = np.ceil(d/2).astype('int32') # choose eigenvectors according the largest eigvalues
        
        # Parameter Initialization
        
        eig, vec = tf.linalg.eigh(S) # eigenvalues are sorted in non-descending order
        eig, vec = eig[:,::-1], vec[:,:,::-1] # reorder eigencalues and eigenvectors in descending direction
        self.sigma2 = tf.square(tf.reduce_mean(eig[:, q:], axis=1)) # noise variance
        U = vec[:, :, :q] # matrix of q eigvectors of covariance matrix
        K = tf.linalg.diag(eig[:, :q]) # diagonal matrix of q largest eigenvalues
        K_centered = tf.sqrt(K - self.sigma2[:, np.newaxis, np.newaxis]*tf.linalg.eye(q))
        self.W = tf.einsum('ikl,ilm->ikm', U, K_centered) 
#         self.W = tf.random.uniform([3, d, q], minval=-0.5)

        
    def res_prob(self, inputs):
        """ calculate an energy of each row of z

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data each row of which is calculated its energy.

        Returns
        -------
        reconstruction probability or anomaly score : tf.Tensor, shape (n_samples)

        """
        codings = self.encoder(inputs)
        recons = self.decoder(codings)
        z = self.extracter((inputs, recons, codings))
        self.d = d = z.shape[-1]
        gamma = self.est_network(z)
        
        # Calculate mu, sigma
        # i   : index of samples
        # k   : index of components
        # l,m : index of features
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        self.phi = phi = tf.reduce_mean(gamma, axis=0)
        self.mu = mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:, np.newaxis]
        z_centered_1 = tf.sqrt(gamma[:, :, np.newaxis]) * (z[:, np.newaxis, :] - mu[np.newaxis, :, :])
        S =  tf.einsum('ikl,ikm->klm', z_centered_1, z_centered_1) / gamma_sum[:, np.newaxis, np.newaxis] 
        q = np.ceil(d/2).astype('int32') # choose eigenvectors according the largest eigvalues
        
        sigma2I = self.sigma2[:, np.newaxis, np.newaxis]*tf.linalg.eye(q)
        M = sigma2I + tf.einsum('ikl,ilm->ikm', tf.transpose(self.W, [0, 2, 1]), self.W)
        Minv = tf.linalg.inv(M)
        SW = tf.einsum('ikl,ilm->ikm', S, self.W) 
        MinvWT = tf.einsum('ikl,ilm->ikm', Minv, tf.transpose(self.W, [0, 2, 1]))
        min_vals_W = tf.linalg.diag(tf.ones(q, dtype=tf.float32)) * 1e-4
        A = self.sigma2[:, np.newaxis, np.newaxis]*tf.linalg.eye(q) + tf.einsum('ikl,ilm->ikm', MinvWT, SW)
        Ainv = tf.linalg.inv(A + min_vals_W[np.newaxis, :, :])
        
        W_new = tf.einsum('ikl,ilm->ikm', SW, Ainv)
        MinvWnT = tf.einsum('ikl,ilm->ikm', Minv, tf.transpose(W_new, [0, 2, 1]))
        sigma2_new = 1/d*tf.linalg.trace(S - tf.einsum('ikl,ilm->ikm', SW, MinvWnT))
        self.W = W_new
        self.sigma2 = sigma2_new
        

        # model covariance
        C = (self.sigma2[:, np.newaxis, np.newaxis]*tf.linalg.eye(d) + 
             tf.einsum('ikl,ilm->ikm', self.W, tf.transpose(self.W, [0, 2, 1])))


        # Calculate a cholesky decomposition of model covariance

        min_vals = tf.linalg.diag(tf.ones(d, dtype=tf.float32)) * 1e-4
        self.L = L = tf.linalg.cholesky(C + min_vals[np.newaxis, :, :])
        z_centered_2 = z[:, np.newaxis, :] - mu[np.newaxis, :, :] 
        v = tf.linalg.triangular_solve(L, tf.transpose(z_centered_2, [1, 2, 0]))  # kli

        # log(det(cov)) = 2 * sum[log(diag(L))]
        log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        logits = tf.math.log(phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_cov[:, np.newaxis])
        res_probs = - tf.reduce_logsumexp(logits, axis=0)
        
        # Calculate the reconstruction error of MMPCA
        
        W2 = tf.einsum('ikl,ilm->ikm', tf.transpose(self.W, [0, 2, 1]), self.W) # calculate W'*W
        min_vals_W2 = tf.linalg.diag(tf.ones(q, dtype=tf.float32)) * 1e-5
        W2inv = tf.linalg.inv(W2 + min_vals_W2)
        WW2inv = tf.einsum('ikl,ilm->ikm', self.W, W2inv)
        #         L_q =  tf.linalg.cholesky(Q + min_vals_q[np.newaxis, :, :]) # using a cholesky decomposition for matrix inverse
        #         v_q = tf.linalg.triangular_solve(L_q, tf.transpose(W, [0, 2, 1]))  
        #         B = tf.einsum('ikl, ilm->ikm', tf.transpose(v_q, [0, 2, 1]), v_q) # transition matrix
        B = tf.einsum('ikl, ilm->ikm', WW2inv, tf.transpose(self.W, [0, 2, 1])) # transition matrix
        z_i = tf.einsum('ikl, ilm->ikm', B, tf.transpose(z_centered_2, [1, 2, 0])) + mu[:, :, np.newaxis]# individual component reconstruction of z 
        z_res = tf.einsum('ik, ikm->im', gamma, tf.transpose(z_i, [2, 0, 1])) # reconstruction of z
        latent_error = tf.reduce_sum(tf.square(z - z_res), axis=1) # reconstruction error of latent variables

        return res_probs, A, C, latent_error


    def fit(self, inputs, X_test, y_test):
        tf.random.set_seed(self.seed)
        np.random.seed(seed=self.seed)
        
        if self.normalize == 'MinMax':
            self.scaler = MinMaxScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
        elif self.normalize == 'Standard':
            self.scaler = StandardScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
            
        X_train, X_valid = train_test_split(inputs, test_size=0.3, random_state=42)

        n_steps = len(X_train) // self.batch_size
        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)
#         optimizer = keras.optimizers.Adamax(learning_rate=self.lr)
        # loss_fn = keras.losses.mean_squared_error
        mean_loss = keras.metrics.Mean(name='mean_loss')
        metrics = keras.metrics.Mean(name='val_loss')
        minimum_val_loss = 1e10
        best_epoch = 1
        best_model = None
        wait = 0

        for epoch in range(1, self.n_epochs + 1):
            print("Epoch {}/{}".format(epoch, self.n_epochs))
            for step in range(1, n_steps + 1):
                X_batch = random_batch(X_train, batch_size=self.batch_size)

                with tf.GradientTape() as tape:
                    main_loss = self.custom_loss(X_batch)
                    loss = tf.add_n([main_loss] + self.damppca.losses)
                gradients = tape.gradient(loss, self.damppca.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.damppca.trainable_variables))
                
                for variable in self.damppca.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                mean_loss(loss)
                print_status_bar(step * self.batch_size, len(inputs), mean_loss)
            val_loss = tf.add_n([self.custom_loss(X_valid)] + self.damppca.losses)
            loss = tf.add_n([self.custom_loss(X_train)] + self.damppca.losses)
            wait +=1
            
            if val_loss < minimum_val_loss - 0.01*abs(minimum_val_loss):
                minimum_val_loss = val_loss
                self.best_epoch = epoch
                self.damppca.save_weights("my_keras_weights.ckpt")
                wait = 0
            
            elif val_loss >= minimum_val_loss - 0.01*abs(minimum_val_loss) and val_loss < minimum_val_loss:
                minimum_val_loss = val_loss
                
            if wait >= self.patience:
                break 
            mean_loss(self.custom_loss(X_batch))
            metrics(val_loss)
            print_status_bar(len(inputs), len(inputs), mean_loss, [metrics])
            for metric in [mean_loss] + [metrics]:
                metric.reset_states()
            print('Wait:', wait) 
            print("Best Epoch: %d" % (best_epoch))
            self.metrics_cal(X_test, y_test)
            self.mu_list.append(self.mu)
            self.phi_list.append(self.phi)
            self.L_list.append(self.L)

#         print('load weights of best epoch:', self.best_epoch)
            if val_loss < minimum_val_loss - 2:
                break
        self.damppca.load_weights("my_keras_weights.ckpt")

    
    def validate(self, inputs):
        # calculate the energy of input samples
        if self.normalize:
            inputs = self.scaler.transform(inputs)
        
        codings = self.encoder(inputs)
        recons = self.decoder(codings)
        z = self.extracter((inputs, recons, codings))
        z_centered = z[:, np.newaxis, :] - self.mu[np.newaxis, :, :] 
        v = tf.linalg.triangular_solve(self.L, tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(cov)) = 2 * sum[log(diag(L))]
        log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        logits = tf.math.log(self.phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + self.d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_cov[:, np.newaxis])
        res_probs = - tf.reduce_logsumexp(logits, axis=0)

        return res_probs.numpy()
    
    def predict(self, inputs):
        # calculate the energy of input samples
        phi = self.phi_list[self.best_epoch-1]
        mu = self.mu_list[self.best_epoch-1]
        L = self.L_list[self.best_epoch-1]
        if self.normalize:
            inputs = self.scaler.transform(inputs)
        
        codings = self.encoder(inputs)
        recons = self.decoder(codings)
        z = self.extracter((inputs, recons, codings))
        z_centered = z[:, np.newaxis, :] - mu[np.newaxis, :, :] 
        v = tf.linalg.triangular_solve(L, tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(cov)) = 2 * sum[log(diag(L))]
        log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        logits = tf.math.log(phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + self.d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_cov[:, np.newaxis])
        res_probs = - tf.reduce_logsumexp(logits, axis=0)

        return res_probs.numpy()
    
    def metrics_cal(self, X_test, y_test):
        y_pred = self.validate(X_test)
        # Energy thleshold to detect anomaly = 80% percentile of energies
        anomaly_energy_threshold = np.percentile(y_pred, 50)
        print(f"Energy thleshold to detect anomaly : {anomaly_energy_threshold:.3f}")
        # Detect anomalies from test data
        y_pred_flag = np.where(y_pred >= anomaly_energy_threshold, 1, 0)
        prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_flag, average="binary")
        print(f" Precision = {prec:.3f}")
        print(f" Recall    = {recall:.3f}")
        print(f" F1-Score  = {fscore:.3f}")

    def restore(self):
        model = self.damppca
        return model
    