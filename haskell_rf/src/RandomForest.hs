{-
  File      :  RandomForest.hs
  Copyright : (c) Michael LeMay, 5/14/19
  Contains my implementation of homework 6
  Souces:
-}

module RandomForest (
  RandomForest(..),
  addDecisionTree,
  RandomForest.predict,
  createNewForest
) where


import DecisionTree
import Data.Vector as DV
import Data.List as DL
import Data.Semigroup
import DataSet
import System.Random


{-A random forest is a collection of decision tress, and the constants
  needed to build and determine that decision tree.-}
data RandomForest = RandomForest {
  trees :: [DecisionTree],
  maxDepth :: Int,
  minPoints :: Int,
  dS :: (DataSet (DV.Vector Double)),
  bootstrapSize :: Int,
  generator :: StdGen
  } deriving (Show)


{-Concatenate two random forests.
  rF1 parameters win out by default, but they should all be
  kept the same.-}
instance Semigroup (RandomForest) where
  (<>) rF1 rF2 = if (dS rF1) == mempty
    then RandomForest {
      trees = (trees rF2),
      maxDepth = (maxDepth rF2),
      minPoints = (minPoints rF2),
      dS = (dS rF2),
      bootstrapSize = (bootstrapSize rF2),
      generator = (generator rF2)
    }
    else RandomForest {
      trees = (trees rF1) <> (trees rF2),
      maxDepth = (maxDepth rF1),
      minPoints = (minPoints rF1),
      dS = (dS rF1),
      bootstrapSize = (bootstrapSize rF1),
      generator = (generator rF1)
    }


{-The default value sets all the constants to negative 1, ds to empty, and
  gives a generic generator. Combining will replace generic values with non-generic
  values.-}
instance Monoid (RandomForest) where
  mempty = RandomForest {
      trees = [],
      maxDepth = -1,
      minPoints = -1,
      dS = mempty,
      bootstrapSize = -1,
      generator = mkStdGen 100
    }


{-Creates a new forest, but without any trained trees-}
createNewForest :: (DataSet (DV.Vector Double)) -> Int -> Int -> Int -> StdGen -> RandomForest
createNewForest dataSet depth points bootstrap g = RandomForest {
    trees = [],
    maxDepth = depth,
    minPoints = points,
    dS = dataSet,
    bootstrapSize = bootstrap,
    generator = g
  }


{-Add decision tree function to the new forest-}
addDecisionTree :: RandomForest -> RandomForest
addDecisionTree rF = let
  (dataSubset, newGen) = randomSubset (dS rF) (generator rF) (bootstrapSize rF) mempty
  !trainedTree = trainDecisionTree (maxDepth rF) (minPoints rF) dataSubset
  newRf = RandomForest {
      trees = [trainedTree],
      maxDepth = (maxDepth rF),
      minPoints = (minPoints rF),
      dS = (dS rF),
      bootstrapSize = (bootstrapSize rF),
      generator = newGen
    }
  in newRf <> rF


{-Predict function. Uses fmap, then folds over each resulting vector of predictions
  using the decision tree calcMajorityLabel function to get the votes.
  Uses the same decision tree function to do this-}
predict :: RandomForest -> (DataSet (DV.Vector Double)) -> DV.Vector Int
predict rF dS = let
  dTPreds = DL.map (DecisionTree.predict dS) (trees rF)
  inds = [0..(DV.length (labels dS) - 1)]
  groupVotes = DL.map (getPredsForRow dTPreds) inds
  predictedLabels = DV.fromList $ DL.map calcMajorityLabel groupVotes
  in predictedLabels


{-Gets the given prediction for a row for the whole forest.s-}
getPredsForRow :: [DV.Vector Int] -> Int -> DV.Vector Int
getPredsForRow rawForAll row = let
  predsForRow = DL.foldl' (accumValues row) [] rawForAll
  in DV.fromList predsForRow


{-Accumulate the values in that row across the list of results-}
accumValues :: Int -> [Int] -> DV.Vector Int -> [Int]
accumValues row accum x = (x ! row) : accum
