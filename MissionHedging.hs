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


standardNormalPDF :: Vector Double -> Double
standardNormalPDF zs = (1 / sqrt ((2 * pi)**(fromIntegral $ Vec.length zs))) * exp (-0.5 * (zs `dot` zs))


multiNormPdf :: Vector Double -> Vector Double -> Matrix Double -> Double
multiNormPdf xs means covars = (exp $ (-0.5) * ((residuals <# inv covars) `dot` residuals)) / (sqrt $ (2 * pi)**k * det covars)
  where residuals = Vec.zipWith (-) xs means
        k = fromIntegral $ Vec.length xs


-- | TODO: I'm adapting this from Maruddani (2018), but I think the formula in
-- that paper is slightly wrong, so I'm using a modified version
multiNormRVs :: Vector Double -> Vector Double -> Matrix Double -> Vector Double
multiNormRVs zScores alphas covars =
  Vec.fromList $ zipWith
  (\alpha row -> alpha - 0.5 * Vec.sum row + (Vec.sum $ Vec.zipWith (\z cov -> z * sqrt cov) zScores row))
  (Vec.toList alphas) covarRows
  where covarRows = toRows covars


integralMaxStdev = 6 :: Double
integralNumDivisions = 9 :: Double
integralTrapezoidWidth = integralMaxStdev / integralNumDivisions


integralPoints :: [Vector Double]
integralPoints = [fromList [x, y, z] | x <- uniPoints, y <- uniPoints, z <- uniPoints, abs x + abs y + abs z < 15]
  where uniPointsAbs = map (* integralTrapezoidWidth) [1..integralNumDivisions]
        uniPoints = map negate uniPointsAbs ++ [0] ++ uniPointsAbs


modelDistribution :: (Vector Double, Matrix Double)
modelDistribution =
  let alphas = fromList [0.08, 0.04, 0.03] :: Vector Double
      sigmas = diagl [0.18, 0.18, 0.02] :: Matrix Double
      hedgeCorrelation = 0.5
      correlations = (3><3)
        [ 1, 0               , 0
        , 0, 1               , hedgeCorrelation
        , 0, hedgeCorrelation, 1
        ] :: Matrix Double
      covariances = (sigmas <> correlations) <> sigmas
  in (alphas, covariances)


-- | Utility function with constant relative risk aversion.
crraUtility :: Double -> Double -> Double
crraUtility 1 money = log money
crraUtility rra money = (money**(1 - rra) - 1) / (1 - rra)


utility :: Double -> Double -> Double
utility wealth co2 = log wealth * co2


expectedUtility :: Vector Double -> Double
expectedUtility weights' =
  sum $ map (
  \zScores ->
    let (alphas, covars) = modelDistribution
        weights = weights' `Vec.snoc` 1  -- dummy weight for co2
        pdf = standardNormalPDF zScores
        scaledAlphas = (Vec.zipWith (*) weights alphas)
        scaledCovars = (3><3) $ List.concat [[covars!i!j * weights!i * weights!j | i <- [0..2]] | j <- [0..2]]
        rvs = multiNormRVs zScores scaledAlphas scaledCovars
        u = utility (exp $ Vec.sum $ Vec.slice 0 2 rvs) (exp $ rvs!2)
        width = integralTrapezoidWidth**3
    in u * pdf * width
  ) integralPoints


gradientDescentIO :: (Vector Double -> Double) -> Vector Double -> Double -> Int -> IO (Vector Double)
gradientDescentIO f initialGuess scale numSteps = go initialGuess numSteps
  where go :: Vector Double -> Int -> IO (Vector Double)
        go x 0 = do
          printInfo x (f x)
          printf " (failed to converge after %d steps)\n" numSteps
          return x
        go x stepsLeft =
          let h = 0.0001
              fx = f x
              gradient = Vec.map (\i -> (f (Vec.accum (+) x [(i, h)]) - fx) / h) $ Vec.fromList [0..(Vec.length x - 1)]
              distance = sqrt $ Vec.sum $ Vec.map (**2) gradient
          in if isNaN distance || distance < 0.0000001
             then do
               printInfo x fx
               printf " (converged after %d steps)\n" (numSteps - stepsLeft + 1)
               return x
             else do
               printInfo x fx
               printf " (step %d)\r" (numSteps - stepsLeft + 1)
               go (Vec.zipWith (+) x $ Vec.map (* scale) gradient) (stepsLeft - 1)

        printInfo x fx = printf "EU(%.4f, %.4f) = %.4f" (x!0) (x!1) fx


main :: IO ()
main = do
  let (means, covars) = modelDistribution

  print $ expectedUtility $ Vec.fromList [2.3, 1.0]
  print $ expectedUtility $ Vec.fromList [2.47, 1.23]
  print $ expectedUtility $ Vec.fromList [2.6, 1.4]
  putStrLn ""
  res <- gradientDescentIO expectedUtility (fromList [0.5, 0.5]) 10 100
  return ()
