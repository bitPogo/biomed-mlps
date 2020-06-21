from biomed.preprocessor.normalizer.normalizer import Normalizer
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
from biomed.preprocessor.normalizer.stemFilter import StemFilter
from biomed.preprocessor.normalizer.stopWordsFilter import StopWordsFilter
from biomed.preprocessor.normalizer.lowerFilter import LowerFilter
from biomed.preprocessor.normalizer.punctuationFilter import PunctuationFilter
from nltk import word_tokenize

class SimpleNormalizer( Normalizer ):
    def __init__( self, Filter ):
        self.__Filter = Filter

    def apply( self, Text: str, Flags: str ) -> str:
        return self._reassemble(
            self.__applyFilters( word_tokenize( Text ), Flags )
        )

    def __applyFilters( self, Tokens: list, Flags: str ) -> list:
        Purge = 0

        for Index in range( 0, len( Tokens ) ):
            Tokens[ Index ] = self.__filter( Tokens[ Index ], Flags )
            if not Tokens[ Index ]:
                Purge += 1

        self.__removeEmptyItems( Tokens, Purge )
        return Tokens

    def __filter( self, Token, Flags ) -> str:
        FilteredToken = Token

        if "l" in Flags:
            FilteredToken = self.__Filter[ "l" ].apply( FilteredToken )

        if "w" in Flags:
            FilteredToken = self.__Filter[ "w" ].apply( FilteredToken )

        if "s" in Flags:
            FilteredToken = self.__Filter[ "s" ].apply( FilteredToken )

        FilteredToken = self.__Filter[ "*" ].apply( FilteredToken )

        return FilteredToken

    def __removeEmptyItems( self, Tokens: list, Counter: int ):
        while Counter > 0:
            Tokens.remove( '' )
            Counter -= 1

    class Factory( NormalizerFactory ):
        __ApplicableFilter = {
            "s": StemFilter.Factory.getInstance(),
            "w": StopWordsFilter.Factory.getInstance(),
            "l": LowerFilter.Factory.getInstance(),
            "*": PunctuationFilter.Factory.getInstance()
        }
        @staticmethod
        def getInstance() -> Normalizer:
            return SimpleNormalizer( SimpleNormalizer.Factory.__ApplicableFilter )
