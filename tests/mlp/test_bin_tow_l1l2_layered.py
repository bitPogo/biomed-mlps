import unittest
from unittest.mock import MagicMock, patch
from keras.models import Sequential
from biomed.mlp.bin_tow_l1l2_layered import Bin2L1L2Layered

class Bin2LayerSpec( unittest.TestCase ):
    @patch( 'biomed.mlp.bin_tow_l1l2_layered.Sequential' )
    def test_it_compiles_the_model( self, MC: MagicMock ):
        Model = MagicMock( spec = Sequential )
        MC.return_value = Model

        Simple = Bin2L1L2Layered( MagicMock() )
        Simple.buildModel( MagicMock() )

        Model.compile.assert_called_once()

    @patch( 'biomed.mlp.bin_tow_l1l2_layered.Sequential' )
    def test_it_returns_the_model_summary( self, MC: MagicMock ):
        Summary = "summary"
        def summarize( print_fn ):
            print_fn( Summary )

        Model = MagicMock( spec = Sequential )
        MC.return_value = Model

        Model.summary.side_effect = summarize

        Simple = Bin2L1L2Layered( MagicMock() )
        self.assertEqual(
            Simple.buildModel( MagicMock() ),
            Summary
        )
