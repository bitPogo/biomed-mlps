import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..', 'biomed', 'preprocessor' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from normalizer.stopWordsFilter import StopWordsFilter
from normalizer.filter import Filter

class StopWordsFilterSpec( unittest.TestCase ):

    def it_is_a_filter( self ):
        MyFilter = StopWordsFilter.Factory.getInstance()
        self.assertTrue( isinstance( MyFilter, Filter ) )

    def it_ignores_non_stop_words( self ):
        MyFilter = StopWordsFilter.Factory.getInstance()
        self.assertEqual(
            "poney",
            MyFilter.apply( "poney" )
        )

    def it_returns_a_empty_string_if_the_given_token_is_a_stopword( self ):
        MyFilter = StopWordsFilter.Factory.getInstance()
        self.assertEqual(
            "",
            MyFilter.apply( "the" )
        )

    def it_removes_stopwords_independent_of_their_case( self ):
        MyFilter = StopWordsFilter.Factory.getInstance()
        self.assertEqual(
            "",
            MyFilter.apply( "ThE" )
        )
