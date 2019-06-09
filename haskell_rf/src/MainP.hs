{-
  File      :  MainP.hs
  Copyright : (c) Michael LeMay, 5/16/19
  Contains my implementation of the final project
  Souces:
  https://stackoverflow.com/questions/33270692/how-to-get-command-line-arguments-in-haskell
  https://stackoverflow.com/questions/30029029/haskell-check-if-string-is-valid-number
  http://hackage.haskell.org/package/base-4.12.0.0/docs/Control-Concurrent.html
  https://stackoverflow.com/questions/28169297/how-to-force-main-thread-to-wait-for-all-its-child-threads-finish-in-haskell
-}

module Main where

import System.IO
import System.Environment
import System.Exit
import System.Random
import Data.List as DL
import Data.List.Split as DLS
import Data.Char as DC
import Data.Vector as DV
import Text.Read
import Data.Maybe
import Data.Semigroup
import Control.Parallel.Strategies as P
import Control.Concurrent as C
import Control.DeepSeq

import DataSet
import DecisionTreeP
import RandomForestP

{-Regular main function, just case matches on the args given in. Throws an
  error if there are issues.-}
main :: IO(ExitCode)
main = do
  args <- getArgs
  putStrLn $ show args
  case args of
    [inputFile, percentageOfData, columnOfPredictor, sizeForest, numThreads] ->
          runRandomForest inputFile (read columnOfPredictor :: Int)
            (read percentageOfData :: Double) (read sizeForest :: Int) (read numThreads :: Int)
    _ -> printErrorAndExit "Unknown input given"


{-Print the error and exit with the correct filename-}
printErrorAndExit :: String -> IO(ExitCode)
printErrorAndExit err = do
  putStrLn (err)
  return (ExitFailure 1)


{-Actually runs the random forest. Checks the acceptability of the data and then
  trains the given number of trees using other functions. Then works out the training
  and testing accuracy and prints them to the terminal.-}
runRandomForest :: String -> Int -> Double -> Int -> Int -> IO(ExitCode)
runRandomForest filename columnOfPredictor percentageOfData sizeForest numThreads = do
  putStrLn "Running random forest"
  fContents <- readFile filename
  generator <- getStdGen
  let acceptableData = checkCsvParallel fContents columnOfPredictor percentageOfData
  putStrLn "Acceptable CSV given"
  let (trainSet, validSet, testSet) = parseCsv fContents columnOfPredictor percentageOfData
  if acceptableData
    then if sizeForest >= numThreads && sizeForest `mod` numThreads == 0 && numThreads > 0 && sizeForest > 0
      then let
        (trainSet, validSet, testSet) = parseCsv fContents columnOfPredictor percentageOfData
        workPerThread = (sizeForest `div` numThreads)
        jobs = [workPerThread | x<-[0..(numThreads - 1)]]
        {-These hyperparameters are just treated as constants here for simplicity,
          since the point is the functional programming not the model optimizations-}
        maxDepth = 30
        minPoints = 15
        bootstrapSize = div (DV.length (labels trainSet) * 2) 3
        treesToBuild = []
        in do
          trainedForest <- trainMThreads trainSet generator maxDepth minPoints bootstrapSize jobs
          putStrLn "Finished training the forest"
          let trainPreds = RandomForestP.predictParallel trainedForest trainSet
          let testPreds = RandomForestP.predictParallel trainedForest testSet
          putStrLn "Finished getting predictions"
          let
            {-Compute prediction accuracy-}
            correctCountTrain = DV.foldl' computePredictionCorrectCount 0 (DV.zip trainPreds (labels trainSet))
            accuracyTrain = (fromIntegral correctCountTrain) / (fromIntegral $ DV.length trainPreds)
            correctCountTest = DV.foldl' computePredictionCorrectCount 0 (DV.zip testPreds (labels testSet))
            accuracyTest = (fromIntegral correctCountTest) / (fromIntegral $ DV.length testPreds)
            in do
              putStrLn ("Training accuracy:" DL.++ (show accuracyTrain))
              putStrLn ("Testing accuracy:" DL.++ (show accuracyTest))
              return ExitSuccess
          else printErrorAndExit "Invalid forest and thread counts given. Num threads must divide evenly into num forests"
        else printErrorAndExit "Invalid .csv format given"


{-Trains a tree given a set of tree counts for each thread to train. The length
  of the list determines the number of threads. Spawns list length - 1 threads
  to do build (value in list) number of trees. Uses a shared mVar that each
  thread updates with new versions of the random forest. Waits for all threads
  to terminate before returning the final result (the trained tree)-}
trainMThreads :: (DataSet (DV.Vector Double)) -> StdGen -> Int ->
  Int -> Int -> [Int] -> IO(RandomForest)
trainMThreads dS gen maxDepth minPoints bootstrapSize numTreesToBuild = do
  result <- newMVar (RandomForestP.createNewForest dS maxDepth minPoints bootstrapSize gen)
  --code inspired from stackoverflow
  let threadTreeCounts = DL.drop 1 numTreesToBuild
  threads <- forM (DV.fromList threadTreeCounts) (\v -> forkThread (threadFunc v result))
  threadFunc (numTreesToBuild !! 0) result
  -- Wait for all of them to finish
  Prelude.mapM_ takeMVar threads
  --putStrLn "All threads finished"
  takeMVar result


{-Thread func adds another decision tree, then recurses
  down to add another. Strictly evaluates the new random
  forest-}
threadFunc :: Int -> MVar RandomForest -> IO ()
threadFunc treesToBuild m = if treesToBuild == 0
  then do
    --putStrLn "Done with thread"
    return ()
  else do
    --putStrLn ("Thread starting evaluation" DL.++ (show treesToBuild))
    currM <- takeMVar m
    let !newF = addDecisionTree currM
    --putStrLn $ show (trees newF)
    putMVar m newF
    (threadFunc (treesToBuild - 1) m)


{-Verify that a given csv is of acceptable format-}
checkCsvParallel :: String -> Int -> Double -> Bool
checkCsvParallel raw columnOfPredictor percentageOfData = let
  totalLines = DL.drop 1 (DL.lines raw) {-First line is usually the variable names-}
  numDLines = floor (fromIntegral (DL.length totalLines))
  validSize = numDLines > 50 {-Need more than 100 entries in data set-}
  dLines = DL.take numDLines totalLines
  dFullySplit = P.parMap P.rdeepseq (\x -> DLS.splitOn "," x) dLines
  allInteger = DL.foldl' checkAllInts True dFullySplit
  classPresent = DL.foldl' (checkAllHaveClass columnOfPredictor) True dFullySplit
  sameColumns = DL.foldl' (checkLinesSameLength (DL.length (dFullySplit !! 0))) True dFullySplit
  in validSize && classPresent && sameColumns


{-Parse the csv, given that it is of a readable format-}
parseCsv :: String -> Int -> Double ->
  ((DataSet (DV.Vector Double)), (DataSet (DV.Vector Double)), (DataSet (DV.Vector Double)))
parseCsv raw columnOfPredictor percentageOfData =
  let
    totalLines = DL.drop 1 (DL.lines raw) {-First line is usually the variable names-}
    numDLines = floor (fromIntegral (DL.length totalLines) * percentageOfData)
    dLines = (DL.take numDLines totalLines)
    dMatrix = DL.map (\x -> DLS.splitOn "," x) dLines
    trainSize = floor ((fromIntegral numDLines) * 0.7)
    validSize = floor ((fromIntegral numDLines) * 0.15) {-Test set is the same size as valid-}
    trainD = DL.take trainSize dMatrix
    rest = DL.drop trainSize dMatrix
    validD = DL.take validSize rest
    testD = DL.drop validSize rest
    trainX = getXsParallel columnOfPredictor trainD
    trainY = getYsParallel columnOfPredictor trainD
    validX = getXsParallel columnOfPredictor validD
    validY = getYsParallel columnOfPredictor validD
    testX = getXsParallel columnOfPredictor testD
    testY = getYsParallel columnOfPredictor testD
    train = DataSet {dat = trainX, labels = trainY}
    valid = DataSet {dat = validX, labels = validY}
    test = DataSet {dat = testX, labels = testY}
    in (train, valid, test)


{-fold function to fold over entire structure-}
computePredictionCorrectCount :: Int -> (Int, Int) -> Int
computePredictionCorrectCount accum (predLabel, actualLabel) = if predLabel == actualLabel
  then accum + 1
  else accum


{-Gets the class from a dataset.-}
getYsParallel :: Int -> [[String]] -> DV.Vector Int
getYsParallel classCol mContents = let
  res = P.parMap P.rdeepseq (\x -> read (x !! classCol) :: Int) mContents
  in DV.fromList res


{-remove class col from the input, and convert it all to a matrix-}
getXsParallel :: Int -> [[String]] -> DV.Vector (DV.Vector Double)
getXsParallel classCol mContents = DV.fromList (P.parMap P.rdeepseq (convRowParallel classCol) mContents)


{-Convert a list of strings into a list of doubles with the classCol removed-}
convRowParallel :: Int -> [String] -> DV.Vector Double
convRowParallel classCol row = let
  rowNoClass = (DL.take classCol row) DL.++ (DL.drop (classCol + 1) row)
  res = P.parMap P.rdeepseq (\x -> read x :: Double) rowNoClass
  in DV.fromList res


{-Check all lines are long enough to have the class-}
checkAllHaveClass :: Int -> Bool -> [String] -> Bool
checkAllHaveClass col accum row = (col < DL.length row) && accum


{-Check that all lines are the same length-}
checkLinesSameLength :: Int -> Bool -> [String] -> Bool
checkLinesSameLength len accum row = (len == DL.length row) && accum


{-Check that every single line can be converted to an int using an individual
  line checker-}
checkAllInts :: Bool -> [String] -> Bool
checkAllInts accum strs =
  accum && (DL.foldl' checkLineIsInts True strs)


{-Check that a single line is entirely convertible to ints-}
checkLineIsInts :: Bool -> String -> Bool
checkLineIsInts accum str =
  case (readMaybe str :: Maybe Double) of
    Just val -> accum && True
    Nothing -> False


{-Helper function drawn from stack overflow. Link sourced above-}
forkThread :: IO () -> IO (MVar ())
forkThread proc = do
    handle <- newEmptyMVar
    _ <- forkFinally proc (\_ -> putMVar handle ())
    return handle
