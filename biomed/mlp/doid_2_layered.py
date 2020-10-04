from keras.models import Sequential
from keras.layers import Dense, Dropout
from biomed.properties_manager import PropertiesManager
from keras.regularizers import l1
from biomed.mlp.model_base import ModelBase

class Doid2Layered( ModelBase ):
    def __init__( self, Properties: PropertiesManager ):
        super( Doid2Layered, self ).__init__( Properties )

    def buildModel( self, Shape: tuple, _: None = None ) -> str:
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
        Model.add( Dense( units = 16, activation ='softmax' ) )

        Model.compile(
            loss="categorical_crossentropy",
            optimizer='sgd',
            metrics=['accuracy']
        )

        self._Model = Model
        return self._summarize()
