import tensorflow_datasets as tfds


def get_dataset_ds(dataset_name='mnist', as_supervised=True, **kwargs):
    # Construct a tf.data.Dataset
    # ds_train, ds_val, ds_test
    return tfds.load(dataset_name, split=['train[:80%]', 'train[80%:]', 'test'], as_supervised=as_supervised, **kwargs)
