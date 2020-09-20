from biomed.vectorizer.selector.selector import Selector
from biomed.vectorizer.selector.linear_vector_selector import LinearVectorSelector
from biomed.properties_manager import PropertiesManager
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import unittest
from unittest.mock import MagicMock, patch, ANY

class LinearVectorSelectorSpec( unittest.TestCase ):
    def setUp( self ):
        self.__PM = PropertiesManager()

    def test_it_is_a_Selector( self ):
        MySelect = LinearVectorSelector( self.__PM )
        self.assertTrue( isinstance( MySelect, Selector ) )

    @patch( 'biomed.vectorizer.selector.linear_vector_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.linear_vector_selector.LinearSVC' )
    def test_it_uses_the_given_class_weights( self, LVModel: MagicMock, ModelSelector: MagicMock ):
        Weights = MagicMock()

        MySelect = LinearVectorSelector( self.__PM )

        MySelect.build( MagicMock(), MagicMock(), Weights )

        LVModel.assert_called_once_with(
            class_weight = Weights,
        )
    @patch( 'biomed.vectorizer.selector.linear_vector_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.linear_vector_selector.LinearSVC' )
    def test_it_builds_the_selector_from_the_tree_classifier( self, LVModel: MagicMock, ModelSelector: MagicMock ):
        MySelect = LinearVectorSelector( self.__PM )

        Model = MagicMock( spec = LinearSVC )
        LVModel.return_value = Model

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )

        ModelSelector.assert_called_once_with( Model, prefit = False, max_features = ANY )

    @patch( 'biomed.vectorizer.selector.linear_vector_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.linear_vector_selector.LinearSVC' )
    def test_it_uses_the_given_amount_of_samples( self, LVModel: MagicMock, ModelSelector: MagicMock ):
        MySelect = LinearVectorSelector( self.__PM )

        self.__PM.selection[ 'amountOfFeatures' ] = 42

        Model = MagicMock( spec = LinearSVC )
        LVModel.return_value = Model

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )

        ModelSelector.assert_called_once_with(
            Model,
            prefit = False,
            max_features = self.__PM.selection[ 'amountOfFeatures' ],
        )

    def test_it_fails_to_return_the_supported_features_if_a_selector_was_not_builded( self ):
        MySelect = LinearVectorSelector( self.__PM )
        with self.assertRaises( RuntimeError, msg = "The selector must be builded before using it" ):
            MySelect.getSupportedFeatures( MagicMock() )

    @patch( 'biomed.vectorizer.selector.linear_vector_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.linear_vector_selector.LinearSVC' )
    def test_it_fetches_the_selected_feature_indicies( self, LVModel: MagicMock, ModelSelector: MagicMock ):
        MySelect = LinearVectorSelector( self.__PM )
        Selector = MagicMock( spec = SelectFromModel )
        ModelSelector.return_value = Selector

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )
        MySelect.getSupportedFeatures( MagicMock() )

        Selector.get_support.assert_called_once_with( indices = True )


    @patch( 'biomed.vectorizer.selector.linear_vector_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.linear_vector_selector.LinearSVC' )
    def test_it_returns_the_filtered_FeatureNames( self, LVModel: MagicMock, ModelSelector: MagicMock ):
        MySelect = LinearVectorSelector( self.__PM )
        Selector = MagicMock( spec = SelectFromModel )
        FeatureNames = [ "a", "b", "c", "d", "e", "f" ]
        AcceptedKeys = [ 1, 2, 4 ]

        ModelSelector.return_value = Selector
        Selector.get_support.return_value = AcceptedKeys

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )
        self.assertListEqual(
            [ "b", "c", "e" ],
            MySelect.getSupportedFeatures( FeatureNames )
        )
