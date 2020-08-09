from pandas import Series
from numpy import array as Array
from biomed.vectorizer.vectorizer import Vectorizer
from biomed.vectorizer.vectorizer import VectorizerFactory
from biomed.vectorizer.selector.selector import Selector
import biomed.services as Services
from biomed.properties_manager import PropertiesManager
from sklearn.feature_extraction.text import TfidfVectorizer

class StdVectorizer( Vectorizer ):
    def __init__( self, Properties: PropertiesManager, Selector: Selector ):
        self.__Properties = Properties
        self.__Selector = Selector
        self.__Vectorizer = None

    def __initializeVectorizer( self ):
         self.__Vectorizer = TfidfVectorizer(
             analyzer = self.__Properties.vectorizing[ 'analyzer' ],
             min_df = self.__Properties.vectorizing[ 'min_df' ],
             max_df = self.__Properties.vectorizing[ 'max_df' ],
             max_features = self.__Properties.vectorizing[ 'max_features' ],
             ngram_range = self.__Properties.vectorizing[ 'ngram_range' ],
             use_idf = self.__Properties.vectorizing[ 'use_idf' ],
             smooth_idf = self.__Properties.vectorizing[ 'smooth_idf' ],
             sublinear_tf = self.__Properties.vectorizing[ 'sublinear_tf' ],
             norm = self.__Properties.vectorizing[ 'norm' ],
             binary = self.__Properties.vectorizing[ 'binary' ],
             dtype = self.__Properties.vectorizing[ 'dtype' ],
        )

    def featureizeTrain( self, Train: Series, Labels: Series ) -> Array:
        self.__initializeVectorizer()
        Features = self.__Vectorizer.fit_transform( Train )
        self.__Selector.build( Features, Labels )
        return self.__Selector.select( Features )

    def featureizeTest( self, Test: Series ) -> Array:
        if not self.__Vectorizer:
            raise RuntimeError( "You must extract trainings feature, before" )
        else:
            return self.__Selector.select(
                self.__Vectorizer.transform( Test )
            )

    class Factory( VectorizerFactory ):
        @staticmethod
        def getInstance():
            return StdVectorizer(
                Services.getService( 'properties', PropertiesManager ),
                Services.getService( 'vectorizer.selector', Selector ),
            )
