from keras.models import Sequential
from keras.layers import Dense, Dropout
from biomed.properties_manager import PropertiesManager
from keras.regularizers import l1
from biomed.mlp.model_base import ModelBase
from biomed.mlp.util.weighted_crossentropy import WeightedCrossentropy

class WeightedDoid7Layered( ModelBase ):
    def __init__( self, Properties: PropertiesManager ):
        super( WeightedDoid7Layered, self ).__init__( Properties )

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
        #hidden layer
        Model.add( Dropout(0.5) )
        Model.add( Dense(
            200,
            activation = 'relu',
            kernel_initializer = 'random_uniform',
            bias_initializer='zero')
        )
        #hidden layer
        Model.add( Dropout(0.5 ) )
        Model.add( Dense(
            200,
            activation = 'relu',
            kernel_initializer = 'random_uniform',
            bias_initializer = 'zero'
        ) )
        #hidden layer
        Model.add( Dropout(0.5 ) )
        Model.add( Dense(
            200,
            activation = 'relu',
            kernel_initializer = 'random_uniform',
            bias_initializer = 'zero'
        ) )
        #hidden layer
        Model.add( Dropout(0.5 ) )
        Model.add( Dense(
            200,
            activation = 'relu',
            kernel_initializer = 'random_uniform',
            bias_initializer = 'zero'
        ) )
        #hidden layer
        Model.add( Dropout(0.5 ) )
        Model.add( Dense(
            200,
            activation = 'relu',
            kernel_initializer = 'random_uniform',
            bias_initializer = 'zero'
        ) )
        Model.add( Dropout(0.1))
        #output layer
        Model.add( Dense( units = 16, activation ='softmax' ) )

        Model.compile(
            loss = WeightedCrossentropy(
                'weighted_categorical_crossentropy',
                Weights,
                'categorical',
            ),
            optimizer='sgd',
            metrics=['accuracy']
        )

        self._Model = Model
        self._Weights = Weights
        self._CustomObjects = { 'WeightedCrossentropy': WeightedCrossentropy }
        return self._summarize()
