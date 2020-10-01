from keras.models import Sequential
from keras.layers import Dense
from biomed.properties_manager import PropertiesManager
from keras.regularizers import l2
from biomed.mlp.model_base import ModelBase

class Bin2L2Layered( ModelBase ):
    def __init__( self, Properties: PropertiesManager ):
        super( Bin2L2Layered, self ).__init__( Properties )

    def buildModel( self, Shape: tuple, _: None = None ) -> str:
        Model = Sequential()
        #input layer
        Model.add(
            Dense(
                units = Shape[ 1 ],
                activity_regularizer= l2( 0.0001 ),
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
        return self._summarize()
