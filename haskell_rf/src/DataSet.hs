{-
  File      :  DataSet.hs
  Copyright : (c) Michael LeMay, 5/14/19
  Contains my implementation of homework 6
  Souces:
-}

module DataSet (
  DataSet(..),
  randomSubset
) where


import Data.Vector as DV
import Data.Semigroup
import System.Random
import Control.Applicative


{-This type represents a matrix (dat), and the corresponding label
  for each row (labls)-}
data DataSet a = DataSet {
      dat :: DV.Vector a,
      labels :: DV.Vector Int
    } deriving (Show, Eq)


{-applies function to zipped together data and labels-}
instance Functor (DataSet) where
    fmap f dS = let
      newD = fmap f (dat dS)
      in DataSet {dat = newD, labels = (labels dS)}

{-applies function to zipped together data and labels-}
instance Applicative (DataSet) where
    pure f = let
      newD = pure f
      in DataSet {dat = newD, labels = DV.empty}
    (<*>) fs dS = let
      newRes = (dat fs) <*> (dat dS)
      in DataSet {dat = newRes, labels = (labels dS)}


{-Combines rows of matrix and corresponding label vectors-}
instance Semigroup (DataSet a) where
  (<>) (DataSet {dat = m1, labels = v1}) (DataSet {dat = m2, labels = v2}) =
    DataSet {dat = m1 <> m2, labels = v1 <> v2}


{-Uses the empty functions for Data.Vector-}
instance Monoid (DataSet a) where
  mempty = DataSet {labels = DV.empty, dat = DV.empty}


{-Gets a random subset of the data. Must be passed in a generator.
  Note that sampling is With Replacement. This is intentional-}
randomSubset :: DataSet a -> StdGen -> Int -> DataSet a -> (DataSet a, StdGen)
randomSubset inSet gen finalLength outSet = if DV.length (labels outSet) == finalLength
  then (outSet, gen)
  else let
    (randomVal, nextGen) = randomR (0, DV.length (labels inSet) - 1) gen
    sampledDat = DV.fromList [(dat inSet) ! randomVal]
    sampledLabel = DV.fromList [(labels inSet) ! randomVal]
    rowSet = DataSet {dat = sampledDat, labels = sampledLabel}
    ! newOutSet = outSet <> rowSet
    in randomSubset inSet nextGen finalLength newOutSet
