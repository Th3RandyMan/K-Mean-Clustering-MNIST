import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances

class ImageDictionary:

    def __init__(self, images:np.ndarray = None, n_clusters:int = -1, patch_size:int = -1, stride:int = -1): # maybe add padding
        self.images = images
        self.n_clusters = n_clusters
        self.patch_size = patch_size
        self.stride = stride
        self.Phi = None
        if images is not None:
            self.fit()

    def __repr__(self):
        return f"ImageDictionary(images={self.images.shape}, n_clusters={self.n_clusters}, patch_size={self.patch_size}, stride={self.stride})" \
                + f"\nDictionary shape: {self.Phi.shape}"
    
    def __str__(self):
        try:
            return f"ImageDictionary(images={self.images.shape}, n_clusters={self.n_clusters}, patch_size={self.patch_size}, stride={self.stride})" \
                + f"\nDictionary shape: {self.Phi.shape}"
        except AttributeError:
            return "ImageDictionary not fitted."
    
    def fit(self, images:np.ndarray = None, n_clusters:int = 100, patch_size:int = 5, stride:int = 1):
        if images is not None:
            self.images = images
        if self.images is None:
            raise ValueError("No images to fit.")
        
        if self.n_clusters == -1:
            self.n_clusters = n_clusters
        if self.patch_size == -1:
            self.patch_size = patch_size
        if self.stride == -1:
            self.stride = stride

        self._build_dictionary()

    def _build_dictionary(self):
        patches = self._flatten(self._patches())
        X = patches.reshape(-1, patches.shape[-1])
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)

        os.environ['OMP_NUM_THREADS'] = '1'  # to avoid warning message
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto', verbose=1)    # This can take about 10-15 minutes
        self.kmeans.fit(X)
        self.Phi = self.kmeans.cluster_centers_ # could remove top 3 lines to save space

    def _patches(self, X:np.ndarray = None):
        if X is None:
            X = self.images

        height, width = X[0].shape
        num_windows = (int((width-self.patch_size)/self.stride)+1, int((height-self.patch_size)/self.stride)+1)
        patches = np.zeros((len(X), np.prod(num_windows), self.patch_size, self.patch_size), dtype=type(X.flat[0]))

        for k, image in enumerate(X):
            for i in range(0, height - self.patch_size + 1, self.stride):
                for j in range(0, width - self.patch_size + 1, self.stride):
                    patches[k, i*num_windows[0]+j] = image[i:i+self.patch_size, j:j+self.patch_size]
        return patches
    
    def _collect_patches(self, X:np.ndarray):
        pass
        
    def _flatten(self, X):
        return X.reshape(*X.shape[0:2], -1)
    
    def _unflatten(self, X):
        return X.reshape(*X.shape[0:-1], self.patch_size, self.patch_size)
    
    def save(self, filename:str = "dictionary.npy"):
        if not os.path.exists("codebooks"):
            os.makedirs("codebooks")
        if "codebooks" not in filename:
            filename = "codebooks/"+filename
        np.save(filename, self.Phi)

    def load(self, filename:str = "dictionary.npy", n_clusters:int = 100, patch_size:int = 5, stride:int = 1):
        if "codebooks" not in filename:
            filename = "codebooks/"+filename

        self.Phi = np.load(filename)

        if self.n_clusters == -1:
            self.n_clusters = n_clusters
        if self.patch_size == -1:
            if patch_size == -1:
                self.patch_size = int(np.sqrt(self.Phi.shape[-1]))
            else:
                self.patch_size = patch_size
        if self.stride == -1:
            self.stride = stride
    
    def plot_dictionary(self, figsize:tuple = (10, 10), figname:str = "dictionary.png"):
        import matplotlib.pyplot as plt
        dict_images = self._unflatten(self.Phi)

        h = np.sqrt(self.Phi.shape[0]).astype(int)
        w = np.ceil(self.Phi.shape[0] / h).astype(int)
        fig, ax = plt.subplots(h, w, figsize=figsize)
        for i in range(h):
            for j in range(w):
                if i*w+j >= dict_images.shape[0]:
                    break
                ax[i, j].imshow(dict_images[i*w+j], cmap='gray')
                ax[i, j].axis('off')
        
        plt.tight_layout()
        try:
            # if in jupyter notebook
            if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                plt.title(figname)
                plt.show()
            else:
                if not os.path.exists("figures"):
                    os.makedirs("figures")
                if "figures" not in filename:
                    filename = "figures/"+filename
                plt.savefig(figname)
                plt.close()
        except NameError:
            if not os.path.exists("figures"):
                os.makedirs("figures")
            if "figures" not in filename:
                filename = "figures/"+filename
            plt.savefig(figname)
            plt.close()

    def reconstruct(self, X:np.ndarray, save:bool = False, filename:str = "reconstructed.npy"):
        if len(X.shape) == 2:
            X = X.reshape(1, *X.shape)  # add batch dimension
        patches = self._transform(X)

        if save:
            np.save(filename, patches)


    def _transform(self, X:np.ndarray):
        patches = self._patches(X)  # Get patches
        patches = self._flatten(patches)    # Flatten for normalization
        shape = patches.shape
        patches = patches.reshape(-1, patches.shape[-1])
        patches -= np.mean(patches, axis=0)
        patches /= np.std(patches, axis=0)

        dist = euclidean_distances(patches, self.Phi)
        new_patches = self.Phi[np.argmin(dist, axis=1)].reshape(shape)
        return self._unflatten(new_patches)

        #patches = self._unflatten(patches)
        
        # for image in patches:
        #     pass    # See if shaping is done correctly...
        # return euclidean_distances(patches, self.Phi)
        

if __name__ == "__main__":
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    if not os.path.exists("dictionary.npy"):
        img_dict = ImageDictionary(x_train)
        img_dict.save()
    else:
        img_dict = ImageDictionary()
        img_dict.load()

    img_dict.plot_dictionary()
    print(img_dict)

    img_dict.reconstruct(x_test, save=True)

    pass

        