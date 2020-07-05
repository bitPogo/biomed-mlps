from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from biomed.properties_manager import PropertiesManager
from biomed.mlp.mlp import MLP
from biomed.mlp.mlp import MLPFactory

class SimpleExtendedFFN( MLP ):
    class Factory( MLPFactory ):
        @staticmethod
        def getInstance( Properties: PropertiesManager ):
            return SimpleExtendedFFN( Properties )

    def __init__( self, Properties: PropertiesManager ):
        super( SimpleExtendedFFN, self ).__init__( Properties )


    def build_mlp_model(self, input_dim, nb_classes):
        Model = Sequential()
        #input layer
        Model.add(
            Dense(
                units=1000,
                input_dim = input_dim,
                activation = "relu",
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros"
            )
        )
        #hidden layer
        Model.add(
            Dense(
                units = 500,
                activation = "sigmoid"
            )
        )
        #output layer
        Model.add( Dense( units = nb_classes, activation ='softmax' ) )

        Rate = 0.1
        Decay = Rate / self._Properties.training_properties[ 'epochs' ]
        Momentum = 0.8

        Sgd = SGD(lr = Rate, momentum = Momentum, decay = Decay, nesterov = False )

        Model.compile(
            loss='mean_squared_error',
            optimizer=Sgd,
            metrics=['accuracy']
        )

        Model.summary()
        self._Model = Model
