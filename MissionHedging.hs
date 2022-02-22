{- |
Module      : MissionHedging
Description :

Maintainer  : Michael Dickens <michael@mdickens.me>
Created     : 2022-02-21

-}
{-# LANGUAGE BangPatterns #-}

module Main where

import Control.Parallel
import qualified Data.List as List
import qualified Data.Vector.Storable as Vec
import Debug.Trace
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.HMatrix
import Prelude hiding ((<>))
import System.Random
import Text.Printf


traceShowSelf x = traceShow x x


multiNormPdf :: Vector Double -> Vector Double -> Matrix Double -> Double
multiNormPdf xs means covars = (exp $ (-0.5) * ((residuals <# inv covars) `dot` residuals)) / (sqrt $ (2 * pi)**k * det covars)
  where residuals = Vec.zipWith (-) xs means
        k = fromIntegral $ Vec.length xs


integralMaxStdev = 6 :: Double
integralNumDivisions = 9 :: Double
integralTrapezoidWidth = integralMaxStdev / integralNumDivisions


integralPoints :: [Vector Double]
integralPoints = [fromList [x, y, z] | x <- uniPoints, y <- uniPoints, z <- uniPoints, abs x + abs y + abs z < 15]
  where uniPointsAbs = map (* integralTrapezoidWidth) [1..integralNumDivisions]
        uniPoints = map negate uniPointsAbs ++ [0] ++ uniPointsAbs


modelDistribution :: (Vector Double, Matrix Double)
modelDistribution =
  let mus = fromList [0.08, 0.04, 0.03] :: Vector Double
      sigmas = diagl [0.18, 0.18, 0.02] :: Matrix Double
      hedgeCorrelation = 0.0
      correlations = (3><3)
        [ 1, 0               , 0
        , 0, 1               , hedgeCorrelation
        , 0, hedgeCorrelation, 1
        ] :: Matrix Double
      covariances = (sigmas <> correlations) <> sigmas
  in (mus, covariances)


-- | Utility function with constant relative risk aversion.
crraUtility :: Double -> Double -> Double
crraUtility 1 money = log money
crraUtility rra money = (money**(1 - rra) - 1) / (1 - rra)


utility :: Double -> Double -> Double -> Double -> Double -> Double
utility numYears hedgeProp marketRet hedgeRet co2Ret = log money -- * co2
  where money = exp $ numYears * ((1 - hedgeProp) * marketRet + hedgeProp * hedgeRet)
        co2 = exp $ numYears * co2Ret


-- | Deterministically compute expected utility using a triple integral with custom-sized trapezoids.
--
-- TODO: Maybe could improve on this by calculating a few moments of the Taylor
-- series, like what Estrada does for GMM
expectedUtility :: Double -> Double
expectedUtility hedgeProp =
  -- TODO: By calculating values from z-scores this way, I'm spending "too much"
  -- time calculating values on the anticorrelated side. It would be more
  -- efficient to have smaller trapezoids in the parts of probability space
  -- where the RVs are more likely to co-occur.
  sum $ map (
  \zScores ->
    let (means, covars) = modelDistribution
        stdevs = Vec.fromList $ map (\i -> sqrt $ covars!i!i) [0, 1, 2]
        rvs = Vec.zipWith (+) means $ Vec.zipWith (*) stdevs zScores
        weights = Vec.fromList [1 - hedgeProp, hedgeProp, 0]
        variance = (weights <# covars) `dot` weights
        pdf = multiNormPdf rvs means covars
        u = (utility 1 hedgeProp (rvs!0) (rvs!1) (rvs!2)) - variance/2
        width = (integralTrapezoidWidth**3 * Vec.product stdevs)
    in u * pdf * width
  ) integralPoints


gradientDescent :: (Double -> Double) -> Double -> Double -> Int -> Double
gradientDescent f x scale stepsLeft
  | stepsLeft == 0 = x
  | otherwise =
      let xg = x + 0.0001
          fx = f x
          gradient = (f xg - fx) / (xg - x)
      in if isNaN gradient || abs gradient < 0.0000001
          then x
          else gradientDescent f (x + scale * gradient) scale (stepsLeft - 1)


gradientDescentIO :: (Double -> Double) -> Double -> Double -> Int -> IO Double
gradientDescentIO f initialGuess scale numSteps = go initialGuess numSteps
  where go :: Double -> Int -> IO Double
        go x 0 = return x
        go x stepsLeft =
          let xg = x + 0.0001
              fx = f x
              gradient = (f xg - fx) / (xg - x)
          in if isNaN gradient || abs gradient < 0.0000001
             then return x
             else do
               printf "EU(%.4f) = %.4f (step %d)\r" x fx (numSteps - stepsLeft + 1)
               go (x + scale * gradient) (stepsLeft - 1)


main :: IO ()
main = do
  let (means, covars) = modelDistribution

  res <- gradientDescentIO expectedUtility 0.5 5 100
  printf "EU(%.4f) = %.4f\n" res (expectedUtility res)
