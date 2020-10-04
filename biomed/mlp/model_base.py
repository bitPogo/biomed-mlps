from biomed.properties_manager import PropertiesManager
from biomed.mlp.mlp import MLP
from biomed.mlp.input_data import InputData
from keras.callbacks import EarlyStopping as Stopper
from keras.callbacks import ModelCheckpoint as Checkpoint
from keras.models import load_model as loadModel
from typing import Union
from uuid import uuid4 as uuid
from os import path as Path
from os import remove as removeFile
import numpy as NP

class ModelBase( MLP ):
    def __init__( self, Properties: PropertiesManager ):
        self._Properties = Properties
        self._Model = None
        self.__Trained = False
        self._Weights = None
        self._CustomObjects = None
        self.__FileName = self.__createFileName()

    def __alignFileName( self, FileName: str ) -> str:
        return FileName.replace( '-', '_' )

    def __createFileName( self ):
        return Path.join(
            Path.dirname( __file__ ),
            '..',
            '..',
            '.cache',
            self.__alignFileName( '{}.h5'.format( uuid() ) ),
        )

    def _summarize( self ):
        Summery = []
        self._Model.summary( print_fn = lambda X: Summery.append( X ) )
        return "\n".join( Summery )

    def __initStopper( self ) -> Stopper:
        return Stopper(
            monitor = 'val_loss',
            mode = 'min',
            verbose = 1,
            patience = self._Properties.training[ 'patience' ]
        )

    def __initCheckpoint( self ) -> Checkpoint:
        return Checkpoint(
            self.__FileName,
            monitor = 'val_accuracy',
            mode = 'max',
            verbose = 1,
            save_best_only = True
        )

    def __cleanSavedModel( self ):
        if Path.exists( self.__FileName ):
            removeFile( self.__FileName )

    def __remapWeights( self ) -> dict:
        Remaped = {}
        Keys = list( self._Weights.keys() )
        for Index in range( 0, len( self._Weights ) ):
            Remaped[ Index ] = self._Weights[ Keys[ Index ] ]

        return Remaped


    def __getWeights( self ) -> Union[ dict, None ]:
        if not self._Weights:
            return None
        else:
            return self.__remapWeights()

    def train( self, X: InputData, Y: InputData ) -> dict:
        print("Training...")
        Hist = self._Model.fit(
            x = X.Training,
            y = Y.Training,
            class_weight = self.__getWeights(),
            shuffle = True,
            epochs = self._Properties.training[ 'epochs' ],
            batch_size = self._Properties.training['batch_size'],
            validation_data = ( X.Validation, Y.Validation ),
            workers = self._Properties.training['workers'],
            use_multiprocessing = self.__isMultiprocessing(),
            callbacks = [
                self.__initStopper(),
                self.__initCheckpoint(),
            ]
        )

        self._Model = loadModel( self.__FileName, custom_objects = self._CustomObjects )
        self.__cleanSavedModel()
        self.__Trained = True

        return Hist.history

    def __verifyTraining( self ):
        if not self.__Trained:
            raise RuntimeError( "The model has not be trained" )

    def getTrainingScore( self, X: InputData, Y: InputData ) -> dict:
        self.__verifyTraining()
        return self._Model.evaluate(
                X.Test,
                Y.Test,
                batch_size = self._Properties.training['batch_size'],
                workers = self._Properties.training['workers'],
                use_multiprocessing = self.__isMultiprocessing(),
                return_dict =  True,
                verbose = 0
        )

    def __isMultiClass( self ) -> bool:
        return self._Properties.classifier == 'doid'

    def __isMultiprocessing( self ):
        return True if self._Properties.training[ "workers" ] > 1 else False

    def __predict( self, ToPredict: NP.array ) -> NP.array:
        return self._Model.predict(
            ToPredict,
            batch_size = self._Properties.training['batch_size'],
            workers = self._Properties.training[ 'workers' ],
            use_multiprocessing = self.__isMultiprocessing()
        )

    def __normalizeBinary( self, Predictions: NP.array ) -> NP.array:
        return NP.where( self.__normalize( Predictions ) < 0.5, 0, 1 )

    def __normalize( self, Predictions: NP.array ) -> NP.array:
        return NP.argmax( Predictions, axis = 1 )

    def predict( self, ToPredict: tuple ) -> NP.array:
        self.__verifyTraining()

        print("Generating test predictions...")
        if self.__isMultiClass():
            return self.__normalize( self.__predict( ToPredict ) )
        else:
            return self.__normalizeBinary( self.__predict( ToPredict ) )
