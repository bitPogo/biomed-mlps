from biomed.evaluator.evaluator import Evaluator
from biomed.evaluator.evaluator import EvaluatorFactory
from biomed.properties_manager import PropertiesManager
from biomed.utils.dir_checker import checkDir, toAbsPath
from biomed.utils.file_writer import FileWriter
import biomed.services as Services
from numpy import array as Array
from pandas import Series, DataFrame
import os as OS
from os import mkdir
from datetime import datetime as Time
from sys import getsizeof as memSize
from sklearn.metrics import f1_score as F1
from sklearn.metrics import classification_report as Reporter
import asyncio

class StdEvaluator( Evaluator ):
    def __init__(
        self,
        Properties: PropertiesManager,
        SimpleWriter: FileWriter,
        JSONWriter: FileWriter,
        CSVWriter: FileWriter
    ):
        self.__Properties = Properties
        self.__Path = None
        self.__SimpleWriter = SimpleWriter
        self.__JSONWriter = JSONWriter
        self.__CSVWriter = CSVWriter
        self.__Time = dict()
        self.__LastScore = None
        self.__LastReport = None
        self.__Steps = []

    def __setPath( self, ShortName: str ):
        self.__Path = OS.path.join(
            self.__Properties.result_dir,
            "{}-{}".format( ShortName, Time.now().strftime( '%Y-%m-%d_%H-%M-%S' ) )
        )

    def __makePathForFile( self, FileName: str ) -> str:
        return OS.path.join( self.__Path, FileName )

    def __writeJSON( self, FileName: str, Content: dict ):
        self.__JSONWriter.write(
            self.__makePathForFile( FileName ),
            Content
        )

    def __writeFile( self, FileName: str, Content: list ):
        self.__SimpleWriter.write(
            self.__makePathForFile( FileName ),
            Content
        )

    def __writeCSV( self, FileName: str, Content: dict ):
        self.__CSVWriter.write(
            self.__makePathForFile( FileName ),
            Content
        )

    def start( self, ShortName: str, Desription ):
        checkDir( toAbsPath( self.__Properties.result_dir ) )
        self.__setPath( ShortName )
        mkdir( self.__Path )
        self.__writeJSON( 'config.json', self.__Properties.toDict() )
        self.__writeFile( 'descr.txt', [ Desription ] )

    def __checkIfIsStarted( self ):
        if not self.__Path:
            raise RuntimeError( "You have to start the Evaluator before caputuring stuff" )

    def __enqueueStep( self, Step ):
        self.__checkIfIsStarted()
        self.__Steps.append( Step )

    def captureStartTime( self ):
        self.__Time[ 'start' ] = int( Time.now().strftime( '%s' ) )

    def capturePreprocessingTime( self ):
        self.__Time[ 'preprocessing' ] = int( Time.now().strftime( '%s' ) )

    def captureVectorizingTime( self ):
        self.__Time[ 'vectorizing' ] = int( Time.now().strftime( '%s' ) )

    def captureTrainingTime( self ):
        self.__Time[ 'training' ] = int( Time.now().strftime( '%s' ) )

    def caputrePredictingTime( self ):
        self.__Time[ 'predicting' ] = int( Time.now().strftime( '%s' ) )

    async def __captureData( self, Train: list, Test: list ):
        self.__writeCSV( 'train.csv', { 'pmid': Train } )
        self.__writeCSV( 'test.csv', { 'pmid': Test } )

    def captureData( self, Train: list, Test: list ):
        self.__enqueueStep( self.__captureData( Train, Test ) )

    async def __capturePreprocessedData( self, TrainDocs: Series, TestDocs: Series ):
        Sizes = {
            'train': memSize( list( TrainDocs ) ),
            'test': memSize( list( TestDocs ) )
        }

        self.__writeCSV( "sizes.csv", Sizes )

    def capturePreprocessedData( self, TrainDocs: Series, TestDocs: Series ):
        self.__enqueueStep( self.__capturePreprocessedData( TrainDocs, TestDocs ) )

    def __makeFrameAndSave(
        self,
        FileName: str,
        Data,
        Columns: list,
        Rows: list = None
    ):
        if not Rows:
            DF = DataFrame(
                Data,
                columns = Columns
            )
        else:
            DF = DataFrame(
                Data,
                columns = Columns,
                index = Rows
            )

        DF.to_csv( FileName )

    async def __captureFeatures(
        self,
        TrainFeatures: tuple,
        TestFeatures: tuple,
        BagOfWords: list
    ):
        self.__makeFrameAndSave(
            self.__makePathForFile( 'trainingFeatures.csv' ),
            TrainFeatures[ 1 ],
            BagOfWords,
            TrainFeatures[ 0 ]
        )

        self.__makeFrameAndSave(
            self.__makePathForFile( 'testFeatures.csv' ),
            TestFeatures[ 1 ],
            BagOfWords,
            TestFeatures[ 0 ]
        )

    def captureFeatures(
        self,
        TrainFeatures: tuple,
        TestFeatures: tuple,
        BagOfWords: list
    ):
        self.__enqueueStep( self.__captureFeatures( TrainFeatures, TestFeatures, BagOfWords ) )

    async def __captureTrainingHistory( self, History: dict ):
        self.__writeCSV(
            self.__makePathForFile( 'trainingHistory.csv' ),
            History
        )

    def captureTrainingHistory( self, History: dict ):
        self.__enqueueStep( self.__captureTrainingHistory( History ) )

    async def __captureEvaluationScore( self, Score: dict ):
        self.__writeCSV(
            self.__makePathForFile( 'evalScore.csv' ),
            Score
        )

    def captureEvaluationScore( self, Score: dict ):
        self.__enqueueStep( self.__captureEvaluationScore( Score ) )

    def __justSavePredictions( self, Predictions: Array, PMIds: list ):
        self.__makeFrameAndSave(
            self.__makePathForFile( 'predictions.csv' ),
            [ PMIds, list( Predictions ) ],
            [ 'pmid', self.__Properties.classifier ]
        )

    def __saveLabeledPredictions( self, Predictions: Array, PMIds: list, Labels: list ):
        self.__makeFrameAndSave(
            self.__makePathForFile( 'predictions.csv' ),
            [ list( Predictions ), Labels ],
            [ 'predicted', 'actual' ],
            PMIds
        )

    async def __capturePredictions( self, Predictions: Array, PMIds: list, Actual: list = None ):
        if not Actual:
            self.__justSavePredictions( Predictions, PMIds )
        else:
            self.__saveLabeledPredictions( Predictions, PMIds, Actual )

    def capturePredictions( self, Predictions: Array, PMIds: list, Actual: list = None ):
        self.__enqueueStep( self.__capturePredictions( Predictions, PMIds, Actual ) )

    def __getMacroAndMicroScore( self, Predicted: Array, Actual: list ) -> tuple:
        return (
            F1(
                y_pred = Predicted,
                y_true = Actual,
                average = 'macro'
            ),
            F1(
                y_pred = Predicted,
                y_true = Actual,
                average = 'micro'
            )
        )

    def __scoreBinary( self, Predictions: Array, Actual: list ) -> list:
        MiMa = self.__getMacroAndMicroScore( Predictions, Actual )
        Score = [
            MiMa[ 0 ],
            MiMa[ 1 ],
            F1(
                y_pred = Predictions,
                y_true = Actual,
                average = 'binary'
            )
        ]

        self.__makeFrameAndSave(
            self.__makePathForFile( 'f1.csv' ),
            Score,
            [ 'macro', 'micro', 'binary' ],
        )

        return Score

    def __scoreMulitClass( self, Predictions: Array, Actual: list ) -> list:
        MiMa = self.__getMacroAndMicroScore( Predictions, Actual )
        Score = [
            MiMa[ 0 ],
            MiMa[ 1 ],
            F1(
                y_pred = Predictions,
                y_true = Actual,
                average = 'samples'
            )
        ]

        self.__makeFrameAndSave(
            self.__makePathForFile( 'f1.csv' ),
            Score,
            [ 'macro', 'micro', 'samples' ],
        )

        return Score

    def __makeReport( self, Predicted: Array, Actual: list, Labels: list ) -> str:
        self.__writeCSV(
            'classReport.csv',
            Reporter(
                y_pred = Predicted,
                y_true = Actual,
                labels = Labels,
                output_dict = True
            )
        )

        return  Reporter(
            y_pred = Predicted,
            y_true = Actual,
            labels = Labels,
            output_dict = False
        )

    async def __score(
        self,
        Predictions: Array,
        Actual: list,
        Labels: list
    ):
        if len( Labels ) == 2:
            self.__LastScore = self.__scoreBinary( Predictions, Actual )
        else:
            self.__LastScore = self.__scoreMulitClass( Predictions, Actual )

        self.__LastReport = self.__makeReport( Predictions, Actual, Labels )

    def score( self, Predictions: Array, Actual: list, Labels: list ):
        self.__enqueueStep( self.__score( Predictions, Actual, Labels ) )

    async def __waitForSteps( self ):
        while self.__Steps:
            await self.__Steps.pop()

    def finalize( self ) -> dict:
        self.__checkIfIsStarted()
        self.__writeCSV( 'time.csv', self.__Time )
        asyncio.get_event_loop().run_until_complete( self.__waitForSteps() )

    class Factory( EvaluatorFactory ):
        @staticmethod
        def getInstance() -> Evaluator:
            return StdEvaluator(
                Properties = Services.getService( 'properties', PropertiesManager ),
                SimpleWriter = Services.getService( 'evaluator.simple', FileWriter ),
                JSONWriter = Services.getService( 'evaluator.json', FileWriter ),
                CSVWriter = Services.getService( 'evaluator.csv', FileWriter ),
            )
