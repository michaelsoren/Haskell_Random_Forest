{-
  File      :  DecisionTreeP.hs
  Copyright : (c) Michael LeMay, 5/14/19
  Contains my implementation of homework 6, parallelized
  Souces:
-}

module DecisionTreeP (
  DecisionTree(..),
  predict,
  newEmptyDecisionTree,
  trainDecisionTree,
  calcMajorityLabel
) where


import DataSet
import Data.Vector as DV
import Data.List as DL
import qualified Data.Map.Strict as Mp
import Control.Parallel as P


{-Decision trees consist of children, a feature to split on, a threshold
  to split with, and leaf information if that is relevant. I decided to use
  binary trees because that is both standard practice and it's much more straightforward
  to implement. Empty is the alternative type used for left and right as defaults.
  However, thanks to the leaf boolean, that will rarely if ever be used.-}
data DecisionTree = DecisionTree {
    left :: DecisionTree,
    right :: DecisionTree,

    {-Values used to do the actual learning-}
    feature :: Int,
    threshold :: Double,

    {-if a leaf, then will be computing the majority class-}
    leaf :: Bool,
    label :: Int
  } | Empty deriving (Show, Eq)


{-Creates a new empty decision tree-}
newEmptyDecisionTree :: DecisionTree
newEmptyDecisionTree = DecisionTree {
  left = Empty,
  right = Empty,
  feature = -1,
  threshold = -1.0,
  leaf = True,
  label = -1
}


{-Takes in an empty decision tree and trains the initial split, then
  relies on a more complicated recursive function from there-}
trainDecisionTree :: Int -> Int -> (DataSet (DV.Vector Double)) -> DecisionTree
trainDecisionTree maxDepth minPoints dS = let
  (bestRow, bestInd, _) = findBestSplit dS
  value = ((dat dS) ! bestRow) ! bestInd
  (!greater, !lesser) = splitAtValueParallel dS bestInd value
  !newL = trainDecisionTreeHelperParallel 1 maxDepth minPoints lesser
  !newR = trainDecisionTreeHelperParallel 1 maxDepth minPoints greater
  res = DecisionTree {
    left = newL,
    right = newR,
    feature = bestInd,
    threshold = value,
    leaf = False,
    label = -1
  }
  in res


{-Recursive helper for tree building. If should be a leaf, stops recursing and
  produces the final result tree. Otherwise, keeps recursing after finding the
  best split possible through exhaustive search-}
trainDecisionTreeHelperParallel :: Int -> Int -> Int -> (DataSet (DV.Vector Double)) -> DecisionTree
trainDecisionTreeHelperParallel currDepth maxDepth minPoints dS = let
  numPoints = DV.length (dat dS)
  (_, allSameClass) = DV.foldl' (\(c, accum) x-> if x == c
    then (c, True && accum)
    else (c, False)) ((labels dS) ! 0, True) (labels dS)
  shouldBeLeaf = (currDepth >= maxDepth) || (numPoints <= minPoints) || allSameClass
  in if shouldBeLeaf
    then let
      majorityLabel = calcMajorityLabel (labels dS)
      in DecisionTree {
        left = Empty,
        right = Empty,
        feature = -1,
        threshold = -1,
        leaf = True,
        label = majorityLabel
      }
    else newL `par` newR `pseq` DecisionTree {
        left = newL,
        right = newR,
        feature = bestInd,
        threshold = value,
        leaf = False,
        label = -1
      } where
        !(bestRow, bestInd, _) = findBestSplit dS
        value = ((dat dS) ! bestRow) ! bestInd
        !(greater, lesser) = splitAtValueParallel dS bestInd value
        newDepth = currDepth + 1
        !newL = trainDecisionTreeHelperParallel newDepth maxDepth minPoints lesser
        !newR = trainDecisionTreeHelperParallel newDepth maxDepth minPoints greater


{-Calculate which label (read integer) occured most commonly in a
  data vector-}
calcMajorityLabel :: DV.Vector Int -> Int
calcMajorityLabel labs = let
  (_, lab) = DV.foldl' (\(m, largest) x -> case (Mp.lookup x m) of
    Just count -> if count + 1 > largest
      then (Mp.adjust (+ 1) x m, x)
      else (Mp.adjust (+ 1) x m, largest)
    Nothing -> if largest == -1
      then ((Mp.insert x 1 m), x)
      else ((Mp.insert x 1 m), largest)) (Mp.empty, -1) labs
  in lab


{-Find the best overall split in the data set.-}
findBestSplit :: (DataSet (DV.Vector Double)) -> (Int, Int, Double)
findBestSplit dS = let
  numRows = DV.length (labels dS)
  rowsToTest = [0..(numRows - 1)]
  res = DL.foldl' (findBestSplitHelper dS) (-1, -1, 999.0) rowsToTest
  in res


{-Find the best split over all the rows. Uses recursive functions to do
   this.-}
findBestSplitHelper :: (DataSet (DV.Vector Double)) -> (Int, Int, Double) -> Int -> (Int, Int, Double)
findBestSplitHelper dS best@(_, _, bGini) row = let
  rowLen = DV.length ((dat dS) ! row)
  inds = [0..(rowLen - 1)]
  (rowInd, rowGini) = findBestSplitRow dS row inds
  res = if rowGini < bGini then (row, rowInd, rowGini) else best
  in res


{-Takes in a dataset and the row and a list of indexes to try.
  Returns the best index found on this row-}
findBestSplitRow :: (DataSet (DV.Vector Double)) -> Int -> [Int] -> (Int, Double)
findBestSplitRow dS row inds = DL.foldl' (findBestSplitRowHelper dS row) (999, 999.0) inds


{-Folds over accumulating the best -}
findBestSplitRowHelper :: (DataSet (DV.Vector Double)) -> Int -> (Int, Double) -> Int -> (Int, Double)
findBestSplitRowHelper dS row best@(_, bGini) ind = let
  value = ((dat dS) ! row) ! ind
  (s1@(DataSet {dat=_, labels=l1}), s2@(DataSet {dat=_, labels=l2})) = splitAtValueParallel dS ind value
  !thisGini = computeGiniIndexAtParallel l1 l2
  newAccum = if thisGini < bGini
    then (ind, thisGini)
    else best
  in newAccum


{-Split a given data set at an index with a threshold. Produces a data set
  of rows greater than or equal to the threshold, and a set of rows less
  than the threshold.-}
splitAtValueParallel :: (DataSet (DV.Vector Double)) -> Int -> Double -> ((DataSet (DV.Vector Double)), (DataSet (DV.Vector Double)))
splitAtValueParallel dS index thresh = lessThan `par` greaterThanOrEqual `pseq` (greaterThanOrEqual, lessThan) where --TODO
  !greaterThanOrEqual = splitDataSet dS index thresh (>=)
  !lessThan = splitDataSet dS index thresh (<)


{-Splits a data set with a function.-}
splitDataSet :: (DataSet (DV.Vector Double)) -> Int -> Double -> (Double -> Double -> Bool) -> (DataSet (DV.Vector Double))
splitDataSet dS index thresh f = let
  (resDat, resLab, _) = DV.foldl' (splitDataFoldF f dS index thresh) (DV.empty, DV.empty, 0) (dat dS)
  res = DataSet.DataSet {
    dat = resDat,
    labels = resLab
  }
  in res


{-Helper function to properly fold over filtering dataset values by the given
  index and threshold-}
splitDataFoldF :: (Double -> Double -> Bool) ->
  (DataSet (DV.Vector Double)) -> Int -> Double ->
  (DV.Vector (DV.Vector Double), DV.Vector Int, Int) ->
  DV.Vector Double ->
  (DV.Vector (DV.Vector Double), DV.Vector Int, Int)
splitDataFoldF f dS index thresh (accumDat, accumLab, count) vec =
  if f (vec ! index) thresh
    then (DV.snoc accumDat vec, DV.snoc accumLab ((labels dS) ! count), count + 1)
    else (accumDat, accumLab, count + 1)


{-Compute the gini index for a given split. Splits can only be binary since
  I'm implementing this with binary trees-}
computeGiniIndexAtParallel :: DV.Vector Int -> DV.Vector Int -> Double
computeGiniIndexAtParallel group1 group2 =
    group1Map `par` group2Map `pseq` (group1MapDiv `par` group2Map `pseq` (gini1 `par` gini2 `pseq` gini1 + gini2)) where
      valueMap = Mp.empty
      g1Size = DV.length group1
      g2Size = DV.length group2
      group1Map = DV.foldl' buildMap valueMap group1
      group2Map = DV.foldl' buildMap valueMap group2
      !group1MapDiv = Mp.map (\x -> (fromIntegral x) / (fromIntegral (DV.length group1))) group1Map
      !group2MapDiv = Mp.map (\x -> (fromIntegral x) / (fromIntegral (DV.length group2))) group2Map
      !gini1 = (1.0 - (Mp.foldl' computeGroupGini 0.0 group1MapDiv)) * ((fromIntegral g1Size) / (fromIntegral (g1Size + g2Size)))
      !gini2 = (1.0 - (Mp.foldl' computeGroupGini 0.0 group2MapDiv)) * ((fromIntegral g2Size) / (fromIntegral (g1Size + g2Size)))


{-Builds up the whole map by incrementing the counters for the
  different classes.-}
buildMap :: Mp.Map Int Int -> Int -> Mp.Map Int Int
buildMap m c = case Mp.lookup c m of
  Just x -> Mp.adjust (+ 1) c m
  Nothing -> Mp.insert c 1 m


{-Compute the gini index ratio sum-}
computeGroupGini :: Double -> Double -> Double
computeGroupGini accum val = accum + (val * val)


{-Call predict on trained trees only. If a leaf, returns the class associated
  with the leaf (stored in the leaf).-}
predict :: (DataSet (DV.Vector Double)) -> DecisionTree -> DV.Vector Int
predict dS tree = DV.map (predictRow tree) (dat dS)


{-Gets the prediction for a given row. Recursive-}
predictRow :: DecisionTree -> DV.Vector Double -> Int
predictRow tree vec = if (leaf tree)
  then (label tree)
  else
    let
      feat = feature tree
      val = vec ! feat
      in if (threshold tree) > val
        then (predictRow (left tree)) vec
        else (predictRow (right tree)) vec
