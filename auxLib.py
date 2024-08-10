import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import random
import os
import time


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Conv2DTranspose


import h5py
from tqdm import tqdm
from datetime import datetime


print("Importing library...")

class SyntheticImagesGen:
    '''
    Some examples on how to use SyntheticImagesGen class.

    Initializing the class:
    data = SyntheticImagesGen(10, training=['ferro','neel','stripe'], L=40)

    Generating synthetic data:
    train_images, train_labels = data.dataGenerator()

    Info:    
    data.Info()
    '''

    def __init__(self, training: list[str]=['all'], L=40):
        if training == ['all']:
            training = ['para', 'ferro', 'neel', 'stripe']
        else:
            training = [element.lower() for element in training]

        self.training = training
        self.L = L

    def spin_gen(self, conf: str):
        '''
        Generates a spin configuration for the given configuration. 
        The resulting data is a tuple of the different configurations each one can have.
        For example, ferromagnetic configurations can be spin up ferromagnetic or spin down ferromagnetic.
        In this case, the tuple will contain two LxL matrices, one per each type.
        '''
        spin_conf = []
        if conf == 'ferro':
            spin_conf = [np.ones((self.L, self.L)).astype(int), 
                        -np.ones((self.L, self.L)).astype(int)]
        elif conf == 'neel':
            spin = np.fromfunction(lambda i, j: (-1)**(i + j + (conf == 4)), 
                                (self.L, self.L)).astype(int)
            spin_conf = [spin, -spin]
        elif conf == 'stripe':
            spin = np.fromfunction(lambda i, j: (-1)**j, (self.L, self.L)).astype(int)
            spin_conf = [spin, -spin, spin.T, -spin.T]
        elif conf == 'para':
            spin_conf = np.random.choice([-1, 1], size=(self.L, self.L)).astype(int)
        return spin_conf
    
    def Info(self, number_configs: int):
        return print(f'Number of configurations: {number_configs}\nTraining: {self.training} \nL: {self.L}\n')

    def dataGenerator(self, number_configs: int):
        ''' 
        Generates synthetic data given a number of configurations and the type of training we want to do.
        If 'all', then it will generate all possible configurations evenly distributed. There are 4 types of configurations.
        If the number of configurations is not divisible by 4, the remaining configurations will be generated in a paramagnetic manner.
        '''
        start_time = time.time()
        print("Generating synthetic data...")
        config_dict = {
            'para': 1,
            'ferro': 2,
            'neel': 2,
            'stripe': 4
        }

        labels_dict = {
            'para': 0,
            'ferro': 1,
            'neel': 2,
            'stripe': 3
        }

        selected_dict = {k: v for k, v in config_dict.items() if k in self.training}

        selected_labels = {k: v for k, v in labels_dict.items() if k in self.training}
        
        total_configs_per_selected = number_configs // len(selected_dict.values())
        remaining_configs = number_configs % len(selected_dict.values())

        train_images = []
        train_labels = []

        if 'para' in self.training:
            for _ in range(total_configs_per_selected):
                train_images.append(self.spin_gen('para'))
                train_labels.append(labels_dict['para'])
            del selected_dict['para'], selected_labels['para']
        
        for conf in selected_dict:
            total_conf = total_configs_per_selected
            extra_configs = total_conf % selected_dict[conf]    
                
            if extra_configs !=0:
                remaining_configs += extra_configs
            
            total_conf = total_conf // selected_dict[conf]

            for _ in range(total_conf):
                train_images.extend(self.spin_gen(conf))
                for _ in range(config_dict[conf]):
                    train_labels.append(labels_dict[conf])

        if 'para' in self.training:
            for _ in range(remaining_configs):
                train_images.append(self.spin_gen('para'))
                train_labels.append(labels_dict['para'])
        else:
            for i in range(remaining_configs):
                index = i % len(self.training)
                train_images.append(self.spin_gen(self.training[index])[0])
                train_labels.append(labels_dict[self.training[index]])

        temp = list(zip(train_images, train_labels))
        random.shuffle(temp)
        train_images, train_labels = zip(*temp)

        print("Done!")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time, "seconds")
        return np.array(train_images), np.array(train_labels)


class loader_and_saver:
    '''
    Some examples on how to use the loader_and_saver class.

    Initializing the class:
    loading_data = loader_and_saver(os.getcwd())

    Saving data: Given some data set, in this case 'train_images'.
    loading_data.saver(train_images)

    Loading simulated images:
    sim_images, temperature = loading_data.simulatedImages(5)

    Loading data from the loader given some os path:
    sim = loading_data.loader(os.path.join(os.getcwd(),'2024-08-10','data_2'))

    Using the checker:
    loader_and_saver.checker(train_images, sim)
    '''
    def __init__(self, path):
        self.path = path
    
    
    def saver(self, data, directory=None, name_of_file='data'):
        if directory is None:
            directory = datetime.now().strftime('%Y-%m-%d')

        if not os.path.exists(directory):
            os.makedirs(directory)

        base_name = name_of_file.strip()
        existing_files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith('.h5')]
        name_suffix = len(existing_files) + 1
        name = f"{base_name}_{name_suffix}.h5"

        file_path = os.path.join(self.path,directory,name)
        
        with h5py.File(file_path, 'w') as f:
            for i, arr in enumerate(tqdm(data, desc="Saving images", unit="array")):
                f.create_dataset(f'array_{i}', data=arr, compression='gzip', compression_opts=9)
        print("Files saved as", file_path)


    def loader(self, file_name):
        name = file_name
        if name[:-3] !='.h5':
            name += '.h5'

        loaded_list = []
        with h5py.File(name, 'r') as f:
            for key in tqdm(sorted(f.keys(), key=lambda x: int(x.split('_')[1])), 
                            desc="Loading arrays", unit="array"):
                loaded_list.append(f[key][:])
        print("Files loaded!")
        return loaded_list


    def checker(original_list, loaded_list):
        data_is_equal = True
        for original, loaded in zip(original_list, loaded_list):
            if not np.array_equal(original, loaded):
                data_is_equal = False
                break
        if data_is_equal:
            print("The original data set and the loaded data set are identical.")
        else:
            print("The original data set and the loaded data set are NOT identical.")


    def simulatedImages(self, index: int):
        print('Loading simulated images...')

        densityIndices = ['055','06', '061', '062', '063', '064', '065', '07', '08', '09','1']

        loadingPath = os.path.join(self.path,'data',f'data_p{densityIndices[index]}')
        
        simImages = self.loader(loadingPath)
        
        temperature = np.arange(0.0, 5.02, 0.02).tolist()
        dens_format = densityIndices[index][:1]+'.'+densityIndices[index][1:]
        print(f'Data of density p = {dens_format} succesfully loaded.')
        
        return simImages, temperature
    

def latticeGraph(squareLattice: list, size=40):
    """
    Generates a graph of square lattices using the given list of square lattices.

    Parameters:
        squareLattice (list): A list of square lattices to be plotted.
        size (int, optional): The size of the square lattices. Defaults to 40.

    Returns:
        None
    """
    cmap1 = ListedColormap(['white', 'gray', 'black'])
    
    numPlots = len(squareLattice)
    rows = int(np.ceil(numPlots / 3))
    cols = min(3, numPlots)  # Ensure cols is at most 3

    fig = plt.figure(figsize=(10*cols, 10*rows))
    gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[1]*cols + [0.05])

    ax = [fig.add_subplot(gs[i, j]) for i in range(rows) for j in range(cols)]
    
    for i, axi in enumerate(ax):
        if i < numPlots:
            im1 = axi.imshow(squareLattice[i], cmap=cmap1,
                            interpolation='nearest', vmin=-1, vmax=1)
            axi.set_xticks(np.arange(-0.5, size, 5))
            axi.set_yticks(np.arange(-0.5, size, 5))
            axi.set_xticklabels([])
            axi.set_yticklabels([])
        else:
            fig.delaxes(axi)  # Remove empty subplots

    # Add colorbar to the right of the entire figure
    cbar_ax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im1, cax=cbar_ax, ticks=[-1, 0, 1])

    # Display the plot
    plt.tight_layout()
    plt.show()
    return


class DenseNeuralNetworkGen:
    def __init__(self, input_shape, num_classes, layers=None):
        self.model = Sequential()
        if layers is None:
            layers = [
                {'type': 'conv', 'filters': 32, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2},
                {'type': 'conv', 'filters': 64, 'kernel_size': 3, 'activation': 'relu', 'pool_size': 2},
                {'type': 'flatten'},
                {'type': 'dense', 'units': 128, 'activation': 'relu'},
                {'type': 'dense', 'units': num_classes, 'activation': 'softmax'}
            ]

        for i, layer in enumerate(layers):
            layer_type = layer['type']
            if layer_type == 'conv':
                if i == 0:
                    self.model.add(Conv2D(layer['filters'], layer['kernel_size'], 
                                        activation=layer['activation'], input_shape=input_shape))
                else:
                    self.model.add(Conv2D(layer['filters'], layer['kernel_size'], activation=layer['activation']))
                if layer.get('pool_size') is not None:
                    self.model.add(MaxPooling2D(pool_size=layer['pool_size']))
            elif layer_type == 'convTranspose':
                if i == 0:
                    self.model.add(Conv2DTranspose(layer['filters'], layer['kernel_size'], 
                                                strides=layer.get('strides', (1, 1)), 
                                                padding=layer.get('padding', 'valid'), 
                                                activation=layer['activation'], input_shape=input_shape))
                else:
                    self.model.add(Conv2DTranspose(layer['filters'], layer['kernel_size'], 
                                                strides=layer.get('strides', (1, 1)), 
                                                padding=layer.get('padding', 'valid'), 
                                                activation=layer['activation']))
            elif layer_type == 'dense':
                if i == 0:
                    self.model.add(Dense(layer['units'], activation=layer['activation'], input_shape=input_shape))
                else:
                    self.model.add(Dense(layer['units'], activation=layer['activation']))
            elif layer_type == 'flatten':
                self.model.add(Flatten())
            elif layer_type == 'dropout':
                self.model.add(Dropout(layer['rate']))
            else:
                layer_class = getattr(__import__('tensorflow.keras.layers'), layer_type)
                if i == 0:
                    self.model.add(layer_class(**{k: v for k, v in layer.items() if k != 'type'}, input_shape=input_shape))
                else:
                    self.model.add(layer_class(**{k: v for k, v in layer.items() if k != 'type'}))

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

    def model_saver(self, name):
        if name[-3:] != 'h5':
            name += '.h5'
        if not os.path.exists('models'):
            os.makedirs('models')
        path = os.path.join(os.getcwd(), 'models', name)
        self.model.save(path)

    def weights_saver(self, name):
        if name[-3:] != 'h5':
            name += '.h5'
        if not os.path.exists('models'):
            os.makedirs('models')
        path = os.path.join(os.getcwd(), 'models')
        self.model.save_weights(os.path.join(path, f'{name}_weights.h5'))

    def load_weights(self, name):
        path = os.path.join(os.getcwd(), 'models')
        self.model.load_weights(os.path.join(path, f'{name}_weights.h5'))

    def summary(self):
        self.model.summary()

print("Library successfully imported")