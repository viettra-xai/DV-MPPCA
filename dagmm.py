import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
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
    def __init__(self, hidden_layer_sizes, activation="elu", kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
        self.hidden= [keras.layers.Dense(size, activation=activation, kernel_initializer=kernel_initializer,
                                         kernel_regularizer=keras.regularizers.l2(0.05)) for size in hidden_layer_sizes[:-1]]
        self.out = keras.layers.Dense(hidden_layer_sizes[-1], activation=activation,
                                      kernel_initializer=kernel_initializer, kernel_regularizer=keras.regularizers.l2(0.01))

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
                                          kernel_regularizer=keras.regularizers.l2(0.05)) 
                       for size in hidden_layer_sizes[:-1][::-1]]
        self.out = keras.layers.Dense(input_size, activation=keras.activations.linear,
                                      kernel_initializer=kernel_initializer, kernel_regularizer=keras.regularizers.l2(0.01))

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
                                          kernel_regularizer=keras.regularizers.l2(0.05))
                               for size in hidden_layer_sizes[:-1]]
        self.out = keras.layers.Dense(hidden_layer_sizes[-1], activation=keras.activations.softmax,
                                      kernel_initializer=kernel_initializer, kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, z):
        for layer in self.hidden:
            z = layer(z)
            z = self.dropout_layer(z)
        output = self.out(z)
        return output
    

class DAGMM:
    """ Deep Autoencoding Gaussian Mixture Model.
    """

    MODEL_FILENAME = "DAGMM_model"
    SCALER_FILENAME = "DAGMM_scaler"

    def __init__(self, comp_hiddens, est_hiddens, input_size, comp_activation,
            est_activation, kernel_initializer, est_dropout_ratio=0.5, n_epochs=1000, batch_size=128, 
                 lambda1=0.1, lambda2=0.0001, learning_rate=0.0001, patience=10, 
                 normalize='Standard', random_seed=42):
        
        self.encoder = Encoder(comp_hiddens, comp_activation, kernel_initializer)
        self.decoder = Decoder(comp_hiddens, input_size, comp_activation, kernel_initializer)
        inputs = keras.layers.Input(input_size, name="input")
        codings = self.encoder(inputs)
        recons = self.decoder(codings)
        self.comp_network = keras.models.Model(inputs=[inputs], outputs=[recons])
        self.extracter = Extracter()
        latents = self.extracter((inputs, recons, codings))
        self.est_network = Estimator(est_hiddens, est_activation, kernel_initializer, est_dropout_ratio)
        gamma = self.est_network(latents)
        self.dagmm = keras.models.Model(inputs=[inputs], outputs=[gamma])

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = learning_rate
        self.normalize = normalize
        self.patience = patience
        self.scaler = None
        self.seed = random_seed


    def custom_loss(self, inputs):
        # calculate the loss function with three loss components
        recons = self.comp_network(inputs)  
        
        # (1) reconstruction loss 
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - recons), axis=1), axis=0)
        
        # (2)generative probability loss
        energies, _, _, sigma = self.energy(inputs)
        energy_loss = self.lambda1 * tf.reduce_mean(energies)
        
        # (3) sigularity penalty
        diag_loss = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(sigma)))
        cov_loss = self.lambda2 * diag_loss
        
        # sum of three loss components
        loss = reconstruction_loss + energy_loss + cov_loss
        
        return loss
    
    def energy(self, inputs):
        """ calculate an energy of each row of z

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data each row of which is calculated its energy.

        Returns
        -------
        energy : tf.Tensor, shape (n_samples)
            calculated energies
        """
        codings = self.encoder(inputs)
        recons = self.decoder(codings)
        z = self.extracter((inputs, recons, codings))
        gamma = self.est_network(z)
        
        # Calculate mu, sigma
        # i   : index of samples
        # k   : index of components
        # l,m : index of features
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        phi = tf.reduce_mean(gamma, axis=0)
        mu =  tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:, np.newaxis]
        z_centered_1 = tf.sqrt(gamma[:, :, np.newaxis]) * (z[:, np.newaxis, :] - mu[np.newaxis, :, :])
        sigma =  tf.einsum('ikl,ikm->klm', z_centered_1, z_centered_1) / gamma_sum[:, np.newaxis, np.newaxis]

        # Calculate a cholesky decomposition of covariance in advance
        n_features = z.shape[-1]
        min_vals = tf.linalg.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-6
        L = tf.linalg.cholesky(sigma + min_vals[np.newaxis, :, :])


        z_centered_2 = z[:, np.newaxis, :] - mu[np.newaxis, :, :]  # ikl
        v = tf.linalg.triangular_solve(L, tf.transpose(z_centered_2, [1, 2, 0]))  # kli

        # log(det(Sigma)) = 2 * sum[log(diag(L))]
        log_det_sigma = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        d = z.shape[1]
        logits = tf.math.log(phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_sigma[:, np.newaxis])
        energies = - tf.reduce_logsumexp(logits, axis=0)
        

        return energies, phi, mu, sigma


    def fit(self, inputs, X_test, y_test):
        tf.random.set_seed(self.seed)
        np.random.seed(seed=self.seed)
        if self.normalize == 'MinMax':
            self.scaler = MinMaxScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
        elif self.normalize == 'Standard':
            self.scaler = StandardScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
            
        
        X_train, X_valid = train_test_split(inputs, test_size=0.1, random_state=42)

        n_steps = len(X_train) // self.batch_size
        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)
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
                    loss = tf.add_n([main_loss] + self.dagmm.losses)
                gradients = tape.gradient(loss, self.dagmm.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.dagmm.trainable_variables))
                for variable in self.dagmm.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                mean_loss(loss)
                print_status_bar(step * self.batch_size, len(inputs), mean_loss)
                
            val_loss = tf.add_n([self.custom_loss(X_valid)] + self.dagmm.losses)
            loss = tf.add_n([self.custom_loss(X_train)] + self.dagmm.losses)
            wait +=1
            
            print('\n wait:', wait)
            if val_loss < minimum_val_loss - 0.001*abs(minimum_val_loss):
                minimum_val_loss = val_loss
                self.best_epoch = epoch
                self.dagmm.save_weights("my_keras_weights.ckpt")
                wait = 0
            elif val_loss >= minimum_val_loss - 0.001*abs(minimum_val_loss) and val_loss < minimum_val_loss:
                minimum_val_loss = val_loss
            
            if wait >= self.patience:
                break 
                
            metrics(val_loss)
            
            print_status_bar(len(inputs), len(inputs), mean_loss, [metrics])
            print("Best Epoch: %d" % (best_epoch))
            for metric in [mean_loss] + [metrics]:
                metric.reset_states()   
            
            self.metrics_cal(X_test, y_test)
        self.dagmm.load_weights("my_keras_weights.ckpt")


    def predict(self, inputs):
        # calculate the energy of input samples
        if self.normalize:
            inputs = self.scaler.transform(inputs)
            
        energies, _, _, _ = self.energy(inputs)

        return energies.numpy()
    
    
    def metrics_cal(self, X_test, y_test):
        y_pred = self.predict(X_test)
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
        model = self.model
        return model


