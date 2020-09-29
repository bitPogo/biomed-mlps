from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import BinaryCrossentropy
from biomed.properties_manager import PropertiesManager
from keras.regularizers import l1
from biomed.mlp.model_base import ModelBase
from biomed.mlp.util.weighted_crossentropy import WeightedCrossentropy

class WeightedBin2LayeredDrop( ModelBase ):
    def __init__( self, Properties: PropertiesManager ):
        super( WeightedBin2LayeredDrop, self ).__init__( Properties )

    def buildModel( self, Shape: tuple, Weights: dict ) -> str:
        Model = Sequential()
        #input layer
        Model.add(
            Dense(
                units = Shape[ 1 ],
                activity_regularizer= l1( 0.0001 ),
                input_dim = Shape[ 1 ],
            )
        )
        Model.add( Dropout( 0.1 ) )
        #output layer
        Model.add( Dense( units = 2, activation ='sigmoid' ) )

        Model.compile(
            loss = WeightedCrossentropy(
                'weighted_binary_crossentropy',
                Weights,
                'bin',
            ),
            optimizer = 'sgd',
            metrics=[ 'accuracy' ]
        )

        self._Weights = Weights
        self._Model = Model
        self._CustomObjects = { 'WeightedCrossentropy': WeightedCrossentropy }
        return self._summarize()
