import unittest
from unittest.mock import MagicMock, patch
from keras.models import Sequential
from biomed.mlp.doid_4_w_layered import WeightedDoid4Layered

class WeightedDoid4LayerSpec( unittest.TestCase ):
    @patch( 'biomed.mlp.doid_4_w_layered.Sequential' )
    def test_it_compiles_the_model( self, MC: MagicMock ):
        Model = MagicMock( spec = Sequential )
        MC.return_value = Model

        Simple = WeightedDoid4Layered( MagicMock() )
        Simple.buildModel( MagicMock(), MagicMock() )

        Model.compile.assert_called_once()

    @patch( 'biomed.mlp.doid_4_w_layered.Sequential' )
    def test_it_returns_the_model_summary( self, MC: MagicMock ):
        Summary = "summary"
        def summarize( print_fn ):
            print_fn( Summary )

        Model = MagicMock( spec = Sequential )
        MC.return_value = Model

        Model.summary.side_effect = summarize

        Simple = WeightedDoid4Layered( MagicMock() )
        self.assertEqual(
            Simple.buildModel( MagicMock(), MagicMock() ),
            Summary
        )
