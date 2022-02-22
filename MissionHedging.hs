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
  let means = fromList [0.08, 0.04, 0.03] :: Vector Double
      stdevs = diagl [0.18, 0.18, 0.02] :: Matrix Double
      hedgeCorrelation = 0.9
      correlations = (3><3)
        [ 1, 0               , 0
        , 0, 1               , hedgeCorrelation
        , 0, hedgeCorrelation, 1
        ] :: Matrix Double
      covariances = (stdevs <> correlations) <> stdevs
  in (means, covariances)


getSample :: Seed -> Matrix Double
getSample seed =
  let (means, covariances) = modelDistribution
  in gaussianSample seed 1 means (trustSym covariances)


-- | Utility function with constant relative risk aversion.
crraUtility :: Double -> Double -> Double
crraUtility 1 money = log money
crraUtility rra money = (money**(1 - rra) - 1) / (1 - rra)


utility :: Double -> Double -> Double -> Double -> Double -> Double
utility numYears hedgeProp marketRet hedgeRet co2Ret =
  if money > 0 then log money * co2
  else log money * co2
  where money = exp $ numYears * ((1 - hedgeProp) * marketRet + hedgeProp * hedgeRet)
        co2 = exp (numYears * co2Ret)


-- | Deterministically compute expected utility using a triple integral with custom-sized trapezoids.
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
        pdf = multiNormPdf rvs means covars
        u = utility 1 hedgeProp (rvs!0) (rvs!1) (rvs!2)
    in u * pdf * (integralTrapezoidWidth**3 * Vec.product stdevs)
  ) integralPoints


simulate :: Double -> Seed -> Double
simulate hedgeProp seed =
  let sample = getSample seed
  in utility 1 hedgeProp (sample!0!0) (sample!0!1) (sample!0!2)


simulationAverage :: Vector Seed -> (Seed -> Double) -> Double
simulationAverage seeds simulate =
  (\vec -> Vec.sum vec / fromIntegral (Vec.length vec)) $ Vec.map simulate seeds


fastOptimize :: IO Double
fastOptimize = go 0.02 1000 0.5 (-99) (-1)
  where go stepSize numSamples prevGuess prevScore direction = do
          gen <- newStdGen
          let seeds = Vec.fromList $ take numSamples $ randoms gen
          let guess = prevGuess + (stepSize * direction)
          let score = simulationAverage seeds (simulate guess)

          -- If still improving, keep moving in the current direction.
          -- If got worse, switch directions and go halfway back.
          -- Basically a binary search
          if score > prevScore
          then go (stepSize * 0.9) (numSamples * 6 `div` 5) guess score direction
          else if abs (prevScore - score) < 0.0001
               then return prevGuess
               else go (stepSize * 0.6) (numSamples * 6 `div` 5) prevGuess prevScore (direction * (-1))


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
  printf "EU(%.4f) = %.4f\r" res (expectedUtility res)
  putStrLn ""


main2 :: IO ()
main2 = do
  let gen = mkStdGen 42069
  gens <- sequence $ [newStdGen, newStdGen, newStdGen, newStdGen, newStdGen]
  let numSamples = 100000
  -- let seeds = Vec.fromList $ take (5 * numSamples) $ randoms gen
  -- let seedBatches = map (\n -> Vec.slice (n * numSamples) numSamples seeds) [0..4]
  let seedBatches = map (\n -> Vec.fromList $ take numSamples $ randoms gen) gens :: [Vec.Vector Seed]
  let hedgeProps = [0.0, 0.1, 0.2, 0.5, 1.0]
  let f seeds hedgeProp = simulationAverage seeds (simulate hedgeProp)

  -- TODO: threading isn't working. I think I need to compile and then set # threads with +RTS
  -- List.foldl' (\x y -> par y x) (print $ zipWith f seedBatches hedgeProps) (zipWith f seedBatches hedgeProps)
  print $ zipWith f seedBatches hedgeProps

  -- res <- fastOptimize
  -- gen <- newStdGen
  -- let seeds = Vec.fromList $ take numSamples $ randoms gen
  -- print res
  -- print $ simulationAverage seeds (simulate res)
