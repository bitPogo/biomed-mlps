from keras.models import Sequential
from keras.layers import Dense
from biomed.properties_manager import PropertiesManager
from keras.regularizers import l1
from biomed.mlp.model_base import ModelBase
from typing import Union

class Bin2Layered( ModelBase ):
    def __init__( self, Properties: PropertiesManager ):
        super( Bin2Layered, self ).__init__( Properties )

    def buildModel( self, Shape: tuple, Weights: Union[ None, dict ] = None ) -> str:
        Model = Sequential()
        #input layer
        Model.add(
            Dense(
                units = Shape[ 1 ],
                activity_regularizer= l1( 0.0001 ),
                input_dim = Shape[ 1 ],
            )
        )
        #output layer
        Model.add( Dense( units = 2, activation ='sigmoid' ) )

        Model.compile(
            loss="binary_crossentropy",
            optimizer='sgd',
            metrics=['accuracy']
        )

        self._Model = Model
        self._Weight = Weights
        return self._summarize()
