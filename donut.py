import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support



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
        self.out_mean = keras.layers.Dense(input_size, activation="linear", kernel_initializer=kernel_initializer,
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
    

class DONUT(keras.Model):
    """ Variational autoencoder - based Anomaly Detection.
    """

    MODEL_FILENAME = "VAE_AnomalyDetector"
    SCALER_FILENAME = "VAE_scaler"

    def __init__(self, input_size, hidden_sizes, activation, kernel_initializer, n_epochs=200,
                 batch_size = 128, learning_rate=0.01, patience=20, normalize='Standard', random_seed=42, **kwargs):
        super().__init__(**kwargs)
        inputs = keras.layers.Input(input_size, name="input")
        self.variational_encoder = Variational_Encoder(hidden_sizes, activation, kernel_initializer)
        self.variational_decoder = Variational_Decoder(hidden_sizes, input_size, activation, kernel_initializer)
        _, _, codings = self.variational_encoder(inputs)
        recons_mean, recons_sigmal, recons = self.variational_decoder(codings)
        self.variational_ae = keras.models.Model(inputs=[inputs], outputs=[recons])
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.normalize = normalize
        
        self.lr = learning_rate
        self.patience = patience
        
        self.seed = random_seed
        self.scaler = None


    def custom_loss(self, inputs):
        
        alpha = tf.where(inputs != 0., 1., 0.) # determine the location of missing and outlier data points
        beta = K.sum(alpha, axis=-1)/tf.cast(tf.size(alpha[0]), dtype='float32')
        
        # latent components: logpz and logqz_x

        codings_mean, codings_log_var, codings = self.variational_encoder(inputs)
        
        logpz = log_normal_pdf(codings, 0., 0.) # log of priority distribution
        
        logqz_x = log_normal_pdf(codings, codings_mean, codings_log_var) # log of approximation posterior
        
        # reconstruction probability logpx_z
        recons_mean, recons_sigmal, _ = self.variational_decoder(codings)
        dist = tfp.distributions.Normal(loc=recons_mean, scale=recons_sigmal, name='Normal') 
        logpx_z = dist.log_prob(inputs)
        
        # sum of elements (apply the equation of the paper)
        ELBO = K.sum(tf.math.multiply(alpha, logpx_z), axis=-1) + beta*K.sum(logpz, axis=-1) - K.sum(logqz_x, axis=-1)
#         ELBO = K.sum(tf.math.multiply(1, logpx_z), axis=-1) + 1*K.sum(logpz, axis=-1) - K.sum(logqz_x, axis=-1)
        
        loss = -K.mean(ELBO)
        
        
        # for numerical check
        
        logpz_mean = K.mean(1*K.sum(logpz, axis=-1))
        logqz_x_mean = K.mean(K.sum(logqz_x, axis=-1))
        logpx_z_mean = K.mean(K.sum(tf.math.multiply(1, logpx_z), axis=-1))  
        
#         print("\n logpz: {} \n logqz_x : {} \n logpx_z: {} \n loss: {}"  .format(K.mean(K.sum(logpz, axis=-1)), 
#                                                                     K.mean(K.sum(logqz_x, axis=-1)), 
#                                                                     K.mean(K.sum(logpx_z, axis=-1)), loss))
        return loss

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
        
        X_train, X_valid = train_test_split(inputs, test_size=0.1, random_state=42)

        n_steps = len(X_train) // self.batch_size

        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)
        mean_loss = keras.metrics.Mean(name='loss')
        metrics = keras.metrics.Mean(name='val_loss')
        self.best_epoch = 1
        best_model = None
        minimum_val_loss = 1e10
        wait = 0

        for epoch in range(1, self.n_epochs + 1):
            injection_matrix = np.random.choice([0, 1], size=X_train.shape, p=[0.0, 1])
            X_train_injected = np.multiply(injection_matrix, X_train)
            print("Epoch {}/{}".format(epoch, self.n_epochs))
            for step in range(1, n_steps + 1):
                X_batch = random_batch(X_train_injected, batch_size=self.batch_size)
                with tf.GradientTape() as tape:
                    main_loss = self.custom_loss(X_batch)
                    loss = tf.add_n([main_loss] + self.variational_ae.losses)

                gradients = tape.gradient(loss, self.variational_ae.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.variational_ae.trainable_variables))
                for variable in self.variational_ae.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                
                mean_loss(loss)
                print_status_bar(step * self.batch_size, len(inputs), mean_loss)
            val_loss = tf.add_n([self.custom_loss(X_valid)] + self.variational_ae.losses)
            loss = tf.add_n([self.custom_loss(X_train)] + self.variational_ae.losses)
            wait +=1
                
            if val_loss < minimum_val_loss - 0.001*abs(minimum_val_loss):
                minimum_val_loss = val_loss
                self.best_epoch = epoch
                self.variational_ae.save_weights("my_keras_weights.ckpt")
                wait = 0
            elif val_loss >= minimum_val_loss - 0.001*abs(minimum_val_loss) and val_loss < minimum_val_loss:
                minimum_val_loss = val_loss
            
            if wait >= self.patience:
                break 
                
            metrics(val_loss)
            print_status_bar(len(inputs), len(inputs), mean_loss, [metrics])
#             print("logpz: %0.2f, logqz_x: %0.2f, logpx_z: %0.2f" %(logpz, logqz_x, logpx_z))
            for metric in [mean_loss] + [metrics]:
                metric.reset_states()
            print('Wait:', wait) 
            print("Best Epoch: %d" % (self.best_epoch))
            self.metrics_cal(X_test, y_test)
        self.variational_ae.load_weights("my_keras_weights.ckpt")
        
    
    
    def predict(self, inputs):
        
        loc_matrix = tf.where(inputs == 0., 0., 1.)
        
        if self.normalize:
            inputs = self.scaler.transform(inputs)
            
        inputs = np.multiply(inputs, loc_matrix)
        # Imputation using MCMC
        M = 10
        mis_loc = tf.where(inputs == 0., 1., 0.)
        X_imputed = inputs
        for i in range(M):
            X_recons = self.variational_ae(X_imputed)
            X_imputed = inputs + tf.math.multiply(X_recons, mis_loc)
            
        # Calculate reconstruction score          
        L = 50
        codings_mean, codings_log_var, _ = self.variational_encoder(X_imputed)
        recons_probability = 0
        for i in range(L):
            codings_normal = tfp.distributions.Normal(loc=codings_mean, scale=K.exp(codings_log_var / 2), name='codings_normal')
            codings = tf.squeeze(codings_normal.sample(1), axis=0)
            recons_mean, recons_sigmal, _ = self.variational_decoder(codings)
            recons_normal = tfp.distributions.Normal(loc=recons_mean, scale=recons_sigmal, name='reconstruction_normal')
#             recons_probability = recons_probability + K.mean(tf.math.multiply(recons_normal.prob(inputs), loc_matrix), axis=-1)
            recons_probability = recons_probability + K.mean(recons_normal.prob(X_imputed), axis=-1)
        anomaly_score = -(recons_probability/L)
        
        return anomaly_score.numpy()
    
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
        return self.variational_ae