from keras.models import Sequential
from keras.layers import Dense
from biomed.properties_manager import PropertiesManager
from keras.regularizers import l1
from biomed.mlp.mlp import MLP
from biomed.mlp.mlp import MLPFactory

class SimpleFFN( MLP ):
    class Factory( MLPFactory ):
        @staticmethod
        def getInstance( Properties: PropertiesManager ):
            return SimpleFFN( Properties )

    def __init__( self, Properties: PropertiesManager ):
        super( SimpleFFN, self ).__init__( Properties )


    def build_mlp_model(self, input_dim, nb_classes):
        Model = Sequential()
        #input layer
        Model.add(
            Dense(
                units=10,
                input_dim = input_dim,
                activation = "relu",
                activity_regularizer = l1(0.0001)
            )
        )
        #hidden layer
        Model.add(
            Dense(
                units = 5,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        #output layer
        Model.add( Dense( units = nb_classes, activation ='sigmoid' ) )

        Model.compile(
            loss='mean_squared_error',
            optimizer='sgd',
            metrics=['accuracy']
        )

        Model.summary()
        self._Model = Model
