import os
from zipfile import ZipFile

import h5py
from openseapy.ml.autoclassifier_tools import MimicHDFFile
from tensorflow.keras.models import Model, load_model


# todo: autoclassifier example
from tensorflow.python.keras.saving.hdf5_format import save_model_to_hdf5, load_model_from_hdf5


class AutoClassifier(Model):
    """
    Autoclassifier class

    This class acts like a tensorflow Model object. The autoclassfier exists of three single tensorflow models,
    encoder, decoder  and classifier. This three models can be designed independently, but output dimension of the
    encoder have to be equal to the input dimensions of the decoder and classifier and the output dimension of the
    decoder have to be equal to the input dimension of the encoder.

    For training the Autoclassifier object can act as an autoencoder or autoclassifier. The mode can be chosen with the
    methods:
       - activate_autoclassifier_mode()
       - activate_autoencoder_mode()
    Then the return of a call of the Autoclassifier object corresponds to the chosen mode.
    """
    def __init__(self, encoder, decoder, classifier, mode='autoclassifier'):
        """
        Parameters
        ----------
        encoder: model
            Encoder model.
        decoder: model or None
            Decoder model. If None, no model will be saved and the mode autoencoder will not be work.
        classifier: model or None
            Classifier model. If None, no model will be saved and the mode autoclassifier will not be work.
        mode: str, optional
            Mode of the object. Default 'autoclassifier'.
        """
        super(AutoClassifier, self).__init__()

        self._mode = None

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

        if mode == 'autoclassifier':
            self.activate_autoclassifier_mode()
        elif mode == 'autoencoder':
            self.activate_autoencoder_mode()
        else:
            raise ValueError('Mode can be only "autoencoder" or "autoclassifier" not "{}"'.format(mode))

    @property
    def mode(self):
        """
        Returns
        -------
        current mode: str
        """
        return self._mode

    def activate_autoclassifier_mode(self):
        """
        Activate autoclassifier mode.
        """
        self._mode = 'autoclassifier'

    def activate_autoencoder_mode(self):
        """
        Activate autoencoder mode.
        """
        self._mode = 'autoencoder'

    def call(self, x):
        """
        Call function of the object.

        Parameters
        ----------
        x: input data

        Returns
        -------
        Evaluated model based on mode.
        """
        if self.mode == 'autoclassifier':
            return self.autoclassify(x)
        elif self.mode == 'autoencoder':
            return self.autoencode(x)
        else:
            raise ValueError('Mode can be only "autoencoder" or "autoclassifier" not "{}"'.format(self.mode))

    def autoencode(self, x):
        """
        Evaluate autencoder.

        Parameters
        ----------
        x: input data

        Returns
        -------
        evaluated model
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)

        return decoded

    def autoclassify(self, x):
        """
        Evaluate autoclassifer.

        Parameters
        ----------
        x: input data

        Returns
        -------
        evaluated model
        """
        encoded = self.encode(x)
        classified = self.classify(encoded)

        return classified

    def encode(self, x):
        """
        Encode data.

        Parameters
        ----------
        x: input data

        Returns
        -------
        encoded data
        """
        if self.encoder is not None:
            return self.encoder(x)
        else:
            raise RuntimeError('Encoder is not defined!')

    def decode(self, x):
        """
        Decode data.

        Parameters
        ----------
        x: input data

        Returns
        -------
        decoded data
        """
        if self.decoder is not None:
            return self.decoder(x)
        else:
            raise RuntimeError('Decoder is not defined!')

    def classify(self, x):
        """
        Classify encoded data

        Parameters
        ----------
        x: input data

        Returns
        -------
        classified data
        """
        if self.classifier is not None:
            return self.classifier(x)
        else:
            raise RuntimeError('Classifier is not defined!')

    def save(self, filepath, overwrite=False, *args, **kwargs):
        """
        Save Autoclassifier object as hdf file.

        Parameters
        ----------
        filepath: String, pathlib.Path object or h5py.File
            path of the file to be saved
        overwrite: bool, optional
            Should an existing file be overwritten? Default True.
        """
        if not isinstance(filepath, h5py.File):
            if not overwrite and os.path.isfile(filepath):
                raise FileExistsError('File exists already. If you want to overwrite, set "overwrite=True".')

            f = h5py.File(filepath, mode='w')
            opened_new_file = True
        else:
            f = filepath
            opened_new_file = False

        try:
            if self.encoder is not None:
                if 'encoder' in list(f.keys()):
                    del f['encoder']
                f_encoder = f.create_group('encoder')

                save_model_to_hdf5(self.encoder, MimicHDFFile(f_encoder), *args, **kwargs)

            if self.decoder is not None:
                if 'decoder' in list(f.keys()):
                    del f['decoder']
                f_decoder = f.create_group('decoder')

                save_model_to_hdf5(self.decoder, MimicHDFFile(f_decoder), *args, **kwargs)

            if self.classifier is not None:
                if 'classifier' in list(f.keys()):
                    del f['classifier']
                f_classifier = f.create_group('classifier')

                save_model_to_hdf5(self.classifier, MimicHDFFile(f_classifier), *args, **kwargs)

            f.flush()
        finally:
            if opened_new_file:
                f.close()

    @classmethod
    def load(cls, filepath, *args, **kwargs):
        """
        Load Autocalssifier object from hdf file.

        Parameters
        ----------
        filepath: String, pathlib.Path object or h5py.File
            path of the file to be loaded

        Returns
        -------
        AutoClassifier object
        """
        opened_new_file = not isinstance(filepath, h5py.File)
        if opened_new_file:
            f = h5py.File(filepath, mode='r')
        else:
            f = filepath

        try:
            if 'encoder' in list(f.keys()):
                encoder = load_model_from_hdf5(MimicHDFFile(f['encoder']), *args, **kwargs)
            else:
                encoder = None

            if 'decoder' in list(f.keys()):
                decoder = load_model_from_hdf5(MimicHDFFile(f['decoder']), *args, **kwargs)
            else:
                decoder = None

            if 'classifier' in list(f.keys()):
                classifier = load_model_from_hdf5(MimicHDFFile(f['classifier']), *args, **kwargs)
            else:
                classifier = None

        finally:
            if opened_new_file:
                f.close()

        return cls(encoder=encoder, decoder=decoder, classifier=classifier)
