import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras



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

    
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean 

class Normal(keras.layers.Layer):
    def call(self, inputs):
        mean, sigmal = inputs
        dist = tfp.distributions.Normal(loc=mean, scale=sigmal, name='Normal') 
        return tf.squeeze(dist.sample(1), axis=0)
    
# Case 1

class Variational_Encoder(keras.Model):
    def __init__(self, encoder_size, activation="elu", kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers= [keras.layers.Dense(size, activation=activation, kernel_initializer=kernel_initializer, 
                                                kernel_regularizer=keras.regularizers.l2(0.01))
                             for size in encoder_size[:-1]]
        self.out_mean = keras.layers.Dense(encoder_size[-1], activation=activation, kernel_initializer=kernel_initializer, 
                                           kernel_regularizer=keras.regularizers.l2(0.01))
        self.out_sigmal = keras.layers.Dense(encoder_size[-1], activation="softplus", kernel_initializer=kernel_initializer, 
                                             kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, x):
        z = keras.layers.Flatten()(x)
        for layer in self.hidden_layers:
            z = layer(z)
        codings_mean = self.out_mean(z)
        codings_sigmal = self.out_sigmal(z) + 1e-4
        codings_log_var = 2*K.log(codings_sigmal)
        codings = Sampling()([codings_mean, codings_log_var]) 
        return codings_mean, codings_log_var, codings
    
class Variational_Decoder(keras.Model):
    # decoder layers of compresion network
    def __init__(self, decoder_size, input_size, activation="elu", kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = [keras.layers.Dense(size,activation=activation, kernel_initializer=kernel_initializer, 
                                                kernel_regularizer=keras.regularizers.l2(0.01))
                              for size in decoder_size[:-1][::-1]]     
        self.out_mean = keras.layers.Dense(input_size, activation=activation, kernel_initializer=kernel_initializer,
                                           kernel_regularizer=keras.regularizers.l2(0.01))
        self.out_sigmal = keras.layers.Dense(input_size, activation="softplus", kernel_initializer=kernel_initializer,
                                             kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, x):
        z = x
        for layer in self.hidden_layers:
            x = layer(x)
        recons_mean = self.out_mean(x)
        recons_sigmal = self.out_sigmal(x) + 1e-4
        recons = Normal()([recons_mean, recons_sigmal])
        return recons_mean, recons_sigmal, recons
    
class Extracter(keras.layers.Layer):
    # Defining the Custom Error Construction Layer between the Input Layer and Output Layer of the Autoencoder
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, x_res, z_c = inputs
        # Calculate Euclid norm, distance
        
        point_loc = tf.where(x==0., 0., 1.)
        x_res = tf.math.multiply(x_res, point_loc)
        
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
    
    

class VAEMPPCA(keras.Model):
    """ Variational autoencoder - based Anomaly Detection.
    """

    MODEL_FILENAME = "VAE_AnomalyDetector"
    SCALER_FILENAME = "VAE_scaler"

    def __init__(self, input_size, comp_hiddens, est_hiddens, comp_activation,est_activation, kernel_initializer, 
                 est_dropout_ratio=0.2,n_epochs=1000, batch_size = 128,
                 lambda1=1, lambda2=1, lambda3=0.5, lambda4=0.0001, learning_rate=0.01, 
                 patience=10, normalize='MinMax', random_seed=42, **kwargs):
        super().__init__(**kwargs)
        inputs = keras.layers.Input(input_size, name="input")
        self.variational_encoder = Variational_Encoder(comp_hiddens, comp_activation, kernel_initializer)
        self.variational_decoder = Variational_Decoder(comp_hiddens, input_size, comp_activation, kernel_initializer)
        _, _, codings = self.variational_encoder(inputs)
        recons_mean, recons_sigmal, recons = self.variational_decoder(codings)
        self.vae = keras.models.Model(inputs=[inputs], outputs=[recons])
        self.extracter = Extracter()
        latents = self.extracter((inputs, recons, codings))
        self.est_network = Estimator(est_hiddens, est_activation, kernel_initializer, est_dropout_ratio)
        gamma = self.est_network(latents)
        self.vaemppca= keras.models.Model(inputs=[inputs], outputs=[gamma])
        
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
        
        ## VAE loss
        # latent components: logpz and logqz_x
        
        alpha = tf.where(inputs != 0., 1., 0.) # determine the location of missing and outlier data points
        beta = K.sum(alpha, axis=-1)/tf.cast(tf.size(alpha[0]), dtype='float32')

        codings_mean, codings_log_var, codings = self.variational_encoder(inputs)
        
        logpz = log_normal_pdf(codings, 0., 0.) # log of priority distribution
        
        logqz_x = log_normal_pdf(codings, codings_mean, codings_log_var) # log of approximation posterior
        
        # reconstruction probability logpx_z
        recons_mean, recons_sigmal, _ = self.variational_decoder(codings)
        recons_normal = tfp.distributions.Normal(loc=recons_mean, scale=recons_sigmal, name='Normal') 
        logpx_z = recons_normal.log_prob(inputs)
        
        # vae loss
        ELBO = K.sum(tf.math.multiply(alpha, logpx_z), axis=-1) + beta*K.sum(logpz, axis=-1) - K.sum(logqz_x, axis=-1)
        
        vae_loss = -self.lambda1 * K.mean(ELBO)
        
        ## MPPCA loss
        
        # (2) reconstruction probability loss
        res_probs, A, C, latent_error = self.res_prob(inputs)
        res_prob_loss = self.lambda2 * tf.reduce_mean(res_probs)
        
        # (3) reconstruction loss of MMPCA 
        pca_loss = self.lambda3 * tf.reduce_mean(latent_error, axis=0)
        
        # (4) sigularity penalty
        diag_loss = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(A))) + tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(C)))
        cov_loss = self.lambda4 * diag_loss
        
        # sum of three loss components
        loss = vae_loss + res_prob_loss + pca_loss + cov_loss
        
#         print("\n logpz: {} \n logqz_x : {} \n logpx_z: {} \n vae_loss: {} \n res_prob_loss: {} \n pca_loss: {}"  
#               .format(K.mean(K.sum(logpz, axis=-1)), K.mean(K.sum(logqz_x, axis=-1)), K.mean(K.sum(logpx_z, axis=-1)), 
#                       vae_loss, res_prob_loss, pca_loss))
        
#         print("\n vae_loss: {} \n res_prob_loss: {}"  
#               .format(vae_loss, res_prob_loss))
        
        
        return loss
    

    
    def para_init(self, inputs): 
        
        loc_matrix = tf.where(inputs == 0., 0., 1.)
        
        if self.normalize=='MinMax':
            self.scaler = MinMaxScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
            
        elif self.normalize=='Standard':
            self.scaler = StandardScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
            
        inputs = np.multiply(inputs, loc_matrix)
            
        codings_mean, codings_log_var, codings = self.variational_encoder(inputs)
        recons_mean, recons_sigmal, recons = self.variational_decoder(codings)
        
        z = self.extracter((inputs, recons, codings))
        self.d = d = z.shape[-1]
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
        codings_mean, codings_log_var, codings = self.variational_encoder(inputs)
        recons_mean, recons_sigmal, recons = self.variational_decoder(codings)
        
#         pt_loc = tf.where(inputs==0., 0., 1.)
#         recons = tf.math.multiply(recons, pt_loc)
        
        z = self.extracter((inputs, recons, codings))
        d = z.shape[-1]
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
        
        loc_matrix = tf.where(inputs == 0., 0., 1.)
        
        if self.normalize == 'MinMax':
            self.scaler = MinMaxScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
        elif self.normalize == 'Standard':
            self.scaler = StandardScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
        
        inputs = np.multiply(inputs, loc_matrix)
        
        X_train, X_valid = train_test_split(inputs, test_size=0.3, random_state=42)

        n_steps = len(X_train) // self.batch_size
        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)
#         optimizer = keras.optimizers.Adamax(learning_rate=self.lr)
        # loss_fn = keras.losses.mean_squared_error
        mean_loss = keras.metrics.Mean(name='mean_loss')
        metrics = keras.metrics.Mean(name='val_loss')
        minimum_val_loss = 1e10
        self.best_epoch = 1
        best_model = None
        wait = 0

        for epoch in range(1, self.n_epochs + 1):
            print("Epoch {}/{}".format(epoch, self.n_epochs))
            for step in range(1, n_steps + 1):
                X_batch = random_batch(X_train, batch_size=self.batch_size)

                with tf.GradientTape() as tape:
                    main_loss = self.custom_loss(X_batch)
                    loss = tf.add_n([main_loss] + self.vaemppca.losses)
                gradients = tape.gradient(loss, self.vaemppca.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.vaemppca.trainable_variables))
                
                for variable in self.vaemppca.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                mean_loss(loss)
                print_status_bar(step * self.batch_size, len(inputs), mean_loss)
            val_loss = tf.add_n([self.custom_loss(X_valid)] + self.vaemppca.losses)
            loss = tf.add_n([self.custom_loss(X_train)] + self.vaemppca.losses)
            wait +=1
            
            if val_loss < minimum_val_loss - 0.01*abs(minimum_val_loss):
                minimum_val_loss = val_loss
                self.best_epoch = epoch
                self.vaemppca.save_weights("my_keras_weights.ckpt")
                wait = 0
            
            elif val_loss >= minimum_val_loss - 0.01*abs(minimum_val_loss) and val_loss < minimum_val_loss:
                minimum_val_loss = val_loss
                
            if wait >= self.patience:
                break 
            mean_loss(loss)
            metrics(val_loss)
            print_status_bar(len(inputs), len(inputs), mean_loss, [metrics])
            for metric in [mean_loss] + [metrics]:
                metric.reset_states()
            print('Wait:', wait) 
            print("Best Epoch: %d" % (self.best_epoch))
            
            self.metrics_cal(X_test, y_test)
            self.mu_list.append(self.mu)
            self.phi_list.append(self.phi)
            self.L_list.append(self.L)
                      
            a, b, c, d= self.for_test(X_train)
#             print("\n VAE-Loss: {} \n MPPCA-Loss-1 : {} \n MPPCA-Loss-2: {} \n Total-Loss: {}" .format(a, b, c, d))

           
            
        self.vaemppca.load_weights("my_keras_weights.ckpt")
        
    
    def for_test(self, inputs):
        
        ## VAE loss
        # latent components: logpz and logqz_x

        alpha = tf.where(inputs != 0., 1., 0.) # determine the location of missing and outlier data points
        beta = K.sum(alpha, axis=-1)/tf.cast(tf.size(alpha[0]), dtype='float32')

        codings_mean, codings_log_var, codings = self.variational_encoder(inputs)
        
        logpz = log_normal_pdf(codings, 0., 0.) # log of priority distribution
        
        logqz_x = log_normal_pdf(codings, codings_mean, codings_log_var) # log of approximation posterior
        
        # reconstruction probability logpx_z
        recons_mean, recons_sigmal, _ = self.variational_decoder(codings)
        recons_normal = tfp.distributions.Normal(loc=recons_mean, scale=recons_sigmal, name='Normal') 
        logpx_z = recons_normal.log_prob(inputs)
        
        # vae loss
        ELBO = K.sum(tf.math.multiply(alpha, logpx_z), axis=-1) + beta*K.sum(logpz, axis=-1) - K.sum(logqz_x, axis=-1)
        
        vae_loss = -self.lambda1 * K.mean(ELBO)
        
        
        ## MPPCA loss
        
        # (2) reconstruction probability loss
        res_probs, A, C, latent_error = self.res_prob(inputs)
        res_prob_loss = self.lambda2 * tf.reduce_mean(res_probs)
        
        # (3) reconstruction loss of MMPCA 
        pca_loss = self.lambda3 * tf.reduce_mean(latent_error, axis=0)
        
        # (4) sigularity penalty
        diag_loss = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(A))) + tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(C)))
        cov_loss = self.lambda4 * diag_loss
        
        # sum of three loss components
        loss = vae_loss + res_prob_loss + pca_loss + cov_loss
  
        
        return  vae_loss, res_prob_loss, cov_loss, loss
        
    
    def validate(self, inputs):
        # calculate the energy of input samples
        
        loc_matrix = tf.where(inputs == 0., 0., 1.)
        
        if self.normalize:
            inputs = self.scaler.transform(inputs)
            
        inputs = np.multiply(inputs, loc_matrix)
            
#         # Imputation using MCMC
#         M = 10
#         mis_loc = tf.where(inputs==0., 1., 0.)
#         X_imputed = inputs
#         for i in range(M):
#             X_recons = self.vae(X_imputed)
#             X_imputed = inputs + tf.math.multiply(X_recons, mis_loc)
            
        # Calculate minus reconstruction probability of VAE         
    
        codings_mean, codings_log_var, codings = self.variational_encoder(inputs)
        recons_probability = 0
        codings_normal = tfp.distributions.Normal(loc=codings_mean, scale=K.exp(codings_log_var / 2), name='codings_normal')
        N = 50
        for i in range(N):
            codings = tf.squeeze(codings_normal.sample(1), axis=0)
            recons_mean, recons_sigmal, _ = self.variational_decoder(codings)
            recons_normal = tfp.distributions.Normal(loc=recons_mean, scale=recons_sigmal, name='reconstruction_normal')
            recons_probability = recons_probability + K.mean(tf.math.multiply(recons_normal.prob(inputs), loc_matrix), axis=-1)
#             recons_probability = recons_probability + K.mean(recons_normal.prob(inputs), axis=-1)
        VAE_anoscore = -(recons_probability/N)
#         VAE_median = tfp.stats.percentile(VAE_anoscore, 50)
        VAE_anoscore_sig = tf.math.sigmoid(VAE_anoscore)
        
        # Calculate minus reconstruction probability of MPPCA
        
        recons_mean, recons_sigmal, recons = self.variational_decoder(codings)
        z = self.extracter((inputs, recons, codings))
        z_centered = z[:, np.newaxis, :] - self.mu[np.newaxis, :, :] 
        v = tf.linalg.triangular_solve(self.L, tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(cov)) = 2 * sum[log(diag(L))]
        log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        logits = tf.math.log(self.phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + self.d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_cov[:, np.newaxis])
        MPPCA_anoscore = - tf.reduce_logsumexp(logits, axis=0)
#         MPPCA_median = tfp.stats.percentile(MPPCA_anoscore, 50)
        MPPCA_anoscore_sig = tf.math.sigmoid(MPPCA_anoscore)
        
        anomaly_score = tf.math.add(VAE_anoscore_sig, MPPCA_anoscore_sig)
#         print ("VAE_anoscore:", tfp.stats.percentile(VAE_anoscore_sig, 50))
#         print("MPPCA_anoscore:", tfp.stats.percentile(MPPCA_anoscore_sig, 50))
#         print("Anoscore:", tfp.stats.percentile(anomaly_score, 50))
        
        return MPPCA_anoscore_sig.numpy()
    
    
    def predict(self, inputs):
        # calculate the energy of input samples
        phi = self.phi_list[self.best_epoch-1]
        mu = self.mu_list[self.best_epoch-1]
        L = self.L_list[self.best_epoch-1]
        
        loc_matrix = tf.where(inputs == 0., 0., 1.)
        
        if self.normalize:
            inputs = self.scaler.transform(inputs)
            
        inputs = np.multiply(inputs, loc_matrix)
            
#         # Imputation using MCMC
#         M = 30
#         mis_loc = tf.where(inputs==0., 1., 0.)
#         X_imputed = inputs
#         for i in range(M):
#             X_recons = self.vae(X_imputed)
#             X_imputed = inputs + tf.math.multiply(X_recons, mis_loc)
            
        # Calculate minus reconstruction probability of VAE         
        codings_mean, codings_log_var, codings = self.variational_encoder(inputs)
        recons_probability = 0
        codings_normal = tfp.distributions.Normal(loc=codings_mean, scale=K.exp(codings_log_var / 2), name='codings_normal')
        N = 50
        for i in range(N):
            codings = tf.squeeze(codings_normal.sample(1), axis=0)
            recons_mean, recons_sigmal, _ = self.variational_decoder(codings)
            recons_normal = tfp.distributions.Normal(loc=recons_mean, scale=recons_sigmal, name='reconstruction_normal')
            recons_probability = recons_probability + K.mean(tf.math.multiply(recons_normal.prob(inputs), loc_matrix), axis=-1)
#             recons_probability = recons_probability + K.mean(recons_normal.prob(inputs), axis=-1)
        VAE_anoscore = -(recons_probability/N)
        
#         VAE_median = tfp.stats.percentile(VAE_anoscore, 50)
        VAE_anoscore_sig = tf.math.sigmoid(VAE_anoscore)
        
        # Calculate minus reconstruction probability of MPPCA
        
        recons_mean, recons_sigmal, recons = self.variational_decoder(codings)
#         pt_loc = tf.where(inputs==0., 0., 1.)
#         recons = tf.math.multiply(recons, pt_loc)

        z = self.extracter((inputs, recons, codings))
        z_centered = z[:, np.newaxis, :] - mu[np.newaxis, :, :] 
        v = tf.linalg.triangular_solve(L, tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(cov)) = 2 * sum[log(diag(L))]
        log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        logits = tf.math.log(phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + self.d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_cov[:, np.newaxis])
        MPPCA_anoscore = - tf.reduce_logsumexp(logits, axis=0)
#         MPPCA_median = tfp.stats.percentile(MPPCA_anoscore, 50)
        MPPCA_anoscore_sig = tf.math.sigmoid(MPPCA_anoscore)
        
        anomaly_score = tf.math.add(VAE_anoscore_sig, MPPCA_anoscore_sig)

        return MPPCA_anoscore_sig.numpy()
    
    def metrics_cal(self, X_test, y_test):
        y_pred = self.validate(X_test)
        # Energy thleshold to detect anomaly = 80% percentile of energies
        anomaly_energy_threshold = np.percentile(y_pred, 50)
        print(f"Energy threshold to detect anomaly : {anomaly_energy_threshold:.3f}")
        # Detect anomalies from test data
        y_pred_flag = np.where(y_pred >= anomaly_energy_threshold, 1, 0)
        prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_flag, average="binary")
        print(f" Precision = {prec:.3f}")
        print(f" Recall    = {recall:.3f}")
        print(f" F1-Score  = {fscore:.3f}")


    def restore(self):
        return self.vae