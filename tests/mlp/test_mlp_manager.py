import unittest
from unittest.mock import MagicMock, patch, ANY
from biomed.mlp.mlp_manager import MLPManager
from biomed.mlp.mlp import MLP
from biomed.properties_manager import PropertiesManager

class MLPManagerSpec( unittest.TestCase ):
    def setUp( self ):
        self.__B2P = patch( 'biomed.mlp.mlp_manager.Bin2Layered', spec = MLP )
        self.__B2L2P = patch( 'biomed.mlp.mlp_manager.Bin2L2Layered', spec = MLP )
        self.__B2L1L2P = patch( 'biomed.mlp.mlp_manager.Bin2L1L2Layered', spec = MLP )

        self.__WB2P = patch( 'biomed.mlp.mlp_manager.WeightedBin2Layered', spec = MLP )
        self.__WB2DP = patch( 'biomed.mlp.mlp_manager.WeightedBin2LayeredDrop', spec = MLP )
        self.__B3P = patch( 'biomed.mlp.mlp_manager.Bin3Layered', spec = MLP )
        self.__WB3P = patch( 'biomed.mlp.mlp_manager.WeightedBin3Layered', spec = MLP )
        self.__B4P = patch( 'biomed.mlp.mlp_manager.Bin4Layered', spec = MLP )
        self.__WB4P = patch( 'biomed.mlp.mlp_manager.WeightedBin4Layered', spec = MLP )

        self.__D2P = patch( 'biomed.mlp.mlp_manager.Doid2Layered', spec = MLP )
        self.__D3P = patch( 'biomed.mlp.mlp_manager.Doid3Layered', spec = MLP )
        self.__D4P = patch( 'biomed.mlp.mlp_manager.Doid4Layered', spec = MLP )
        self.__D5P = patch( 'biomed.mlp.mlp_manager.Doid5Layered', spec = MLP )
        self.__D6P = patch( 'biomed.mlp.mlp_manager.Doid6Layered', spec = MLP )
        self.__D7P = patch( 'biomed.mlp.mlp_manager.Doid7Layered', spec = MLP )
        self.__D8P = patch( 'biomed.mlp.mlp_manager.Doid8Layered', spec = MLP )
        self.__D9P = patch( 'biomed.mlp.mlp_manager.Doid9Layered', spec = MLP )

        self.__WD3P = patch( 'biomed.mlp.mlp_manager.WeightedDoid3Layered', spec = MLP )
        self.__WD4P = patch( 'biomed.mlp.mlp_manager.WeightedDoid4Layered', spec = MLP )
        self.__WD5P = patch( 'biomed.mlp.mlp_manager.WeightedDoid5Layered', spec = MLP )
        self.__WD6P = patch( 'biomed.mlp.mlp_manager.WeightedDoid6Layered', spec = MLP )
        self.__WD7P = patch( 'biomed.mlp.mlp_manager.WeightedDoid7Layered', spec = MLP )
        self.__WD8P = patch( 'biomed.mlp.mlp_manager.WeightedDoid8Layered', spec = MLP )
        self.__WD9P = patch( 'biomed.mlp.mlp_manager.WeightedDoid9Layered', spec = MLP )

        self.__B2 = self.__B2P.start()
        self.__B2L2 = self.__B2L2P.start()
        self.__B2L1L2 = self.__B2L1L2P.start()

        self.__WB2 = self.__WB2P.start()
        self.__WB2D = self.__WB2DP.start()
        self.__B3 = self.__B3P.start()
        self.__WB3 = self.__WB3P.start()
        self.__B4 = self.__B4P.start()
        self.__WB4 = self.__WB4P.start()

        self.__D2 = self.__D2P.start()
        self.__D3 = self.__D3P.start()
        self.__D4 = self.__D4P.start()
        self.__D5 = self.__D5P.start()
        self.__D6 = self.__D6P.start()
        self.__D7 = self.__D7P.start()
        self.__D8 = self.__D8P.start()
        self.__D9 = self.__D9P.start()

        self.__WD3 = self.__WD3P.start()
        self.__WD4 = self.__WD4P.start()
        self.__WD5 = self.__WD5P.start()
        self.__WD6 = self.__WD6P.start()
        self.__WD7 = self.__WD7P.start()
        self.__WD8 = self.__WD8P.start()
        self.__WD9 = self.__WD9P.start()

        self.__ReferenceModel = MagicMock( spec = MLP )
        self.__B2.return_value = self.__ReferenceModel

    def tearDown( self ):
        self.__B2P.stop()
        self.__B2L2P.stop()
        self.__B2L1L2P.stop()

        self.__WB2P.stop()
        self.__WB2DP.stop()
        self.__B3P.stop()
        self.__WB3P.stop()
        self.__B4P.stop()
        self.__WB4P.stop()

        self.__D2P.stop()
        self.__D3P.stop()
        self.__D4P.stop()
        self.__D5P.stop()
        self.__D6P.stop()
        self.__D7P.stop()
        self.__D8P.stop()
        self.__D9P.stop()

        self.__WD3P.stop()
        self.__WD4P.stop()
        self.__WD5P.stop()
        self.__WD6P.stop()
        self.__WD7P.stop()
        self.__WD8P.stop()
        self.__WD9P.stop()

    def __fakeLocator( self, _, __ ):
        PM = PropertiesManager()
        PM.model = "b2"
        return PM

    def test_it_is_a_mlp_instance( self ):
        self.assertTrue( isinstance( MLPManager.Factory.getInstance( self.__fakeLocator ), MLP ) )

    def test_it_initializes_a_models( self  ):
        Models = {
            "b2": self.__B2,
            "b2l2": self.__B2L2,
            "b2l1l2": self.__B2L1L2,
            "wb2": self.__WB2,
            "wb2d": self.__WB2D,
            "b3": self.__B3,
            "wb3": self.__WB3,
            "b4": self.__B4,
            "wb4": self.__WB4,
            "doid2": self.__D2,
            "doid3": self.__D3,
            "doid4": self.__D4,
            "doid5": self.__D5,
            "doid6": self.__D6,
            "doid7": self.__D7,
            "doid8": self.__D8,
            "doid9": self.__D9,
            "wdoid3": self.__WD3,
            "wdoid4": self.__WD4,
            "wdoid5": self.__WD5,
            "wdoid6": self.__WD6,
            "wdoid7": self.__WD7,
            "wdoid8": self.__WD8,
            "wdoid9": self.__WD9,
        }

        for ModelKey in Models:
            pm = PropertiesManager()
            pm.model = ModelKey

            def fakeLocator( _, __ ):
                return pm

            ServiceGetter = MagicMock()
            ServiceGetter.side_effect = fakeLocator

            MyManager = MLPManager.Factory.getInstance( ServiceGetter )
            MyManager.buildModel( 2 )

            Models[ ModelKey ].assert_called_once_with( pm )
            ServiceGetter.assert_called_once()

    def test_it_deligates_the_dimensionality_to_the_model( self ):
        InputShape = ( 2, 3 )

        MyManager = MLPManager.Factory.getInstance( self.__fakeLocator )
        MyManager.buildModel( InputShape )

        self.__ReferenceModel.buildModel.assert_called_once_with( InputShape, ANY )

    def test_it_deligates_the_training_arguments_without_weights_to_the_model_by_default( self ):
        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3 ) )

        self.__ReferenceModel.buildModel.assert_called_once_with( ANY, None )

    def test_it_deligates_given_weights_to_the_model( self ):
        Weights = MagicMock()

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3 ), Weights )

        self.__ReferenceModel.buildModel.assert_called_once_with( ANY, Weights )

    def test_it_returns_the_summary_of_the_builded_model( self ):
        Expected = "summary"

        self.__ReferenceModel.buildModel.return_value = Expected

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )

        self.assertEqual(
            Expected,
            Model.buildModel( MagicMock() )
        )

    def test_it_returns_the_history_of_the_training( self ):
        Expected = "this should be not a string in real"

        self.__ReferenceModel.train.return_value = Expected

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3 ) )

        self.assertEqual(
            Expected,
            Model.train( MagicMock(), MagicMock() )
        )

    def test_it_returns_the_score_of_the_training( self ):
        Expected = "this should be not a string in real"

        self.__ReferenceModel.getTrainingScore.return_value = Expected

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3 ) )

        self.assertEqual(
            Expected,
            Model.getTrainingScore( MagicMock(), MagicMock() )
        )

    def test_it_returns_the_predictions( self ):
        Expected = "this should be not a string in real"

        self.__ReferenceModel.predict.return_value = Expected

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3) )

        self.assertEqual(
            Expected,
            Model.predict( MagicMock() )
        )
