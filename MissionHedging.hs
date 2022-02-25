{- |
Module      : MissionHedging
Description :

Maintainer  : Michael Dickens <michael@mdickens.me>
Created     : 2022-02-21

-}
{-# LANGUAGE BangPatterns #-}

module Main where

import Control.Parallel
import Data.List as List
import qualified Data.Vector.Storable as Vec
import Debug.Trace
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.HMatrix
import Prelude hiding ((<>))
import System.Random
import Text.Printf


traceShowSelf x = traceShow x x


updateAt :: Int -> (a -> a) -> [a] -> [a]
updateAt i f xs = go i xs
  where go 0 (x:xs) = (f x) : xs
        go j (x:xs) = x : (go (j-1) xs)


standardNormalPDF :: [Double] -> Double
standardNormalPDF zs = (1 / sqrt ((2 * pi)**(fromIntegral $ length zs))) * exp (-0.5 * (sum $ map (**2) zs))


multiNormPDF :: [Double] -> [Double] -> [[Double]] -> Double
multiNormPDF xs means covars =
  (exp $ (-0.5) * ((residuals <# inv covMat) `dot` residuals))
  /
  (sqrt $ (2 * pi)**k * det covMat)
  where residuals = Vec.fromList $ zipWith (-) xs means
        k = fromIntegral $ length xs
        covMat = fromLists covars


multiNormRVs :: [Double] -> [Double] -> [[Double]] -> [Double]
multiNormRVs zScores alphas covars =
  -- TODO: What if covariance is negative?

  -- TODO: expectedUtility never changes when I change the 3rd term. It looks
  -- like the 3rd term is symmetric so it cancels out over the full integral,
  -- and the first 2 terms happen to be the expected utility with U = log w, so
  -- it still works with log utility.
  zipWith
  (\alpha row -> alpha - 0.5 * sum row + (sum $ zipWith (\z cov -> z * sqrt cov) zScores row))
  -- (\alpha row -> alpha - 0.5 * sum row)
  alphas covars


-- https://stats.stackexchange.com/questions/214997/multivariate-log-normal-probabiltiy-density-function-pdf
-- https://math.stackexchange.com/questions/267267/intuitive-proof-of-multivariable-changing-of-variables-formula-jacobian-withou
lognormPDF :: [Double] -> [Double] -> [[Double]] -> Double
lognormPDF ys mus covars =
  exp ((-0.5) * ((residuals <# inv covMat) `dot` residuals))
  * (1 / product ys)
  / sqrt ((2 * pi)**k * (det covMat))
  where k = fromIntegral $ length ys
        covMat = fromLists covars
        residuals = Vec.fromList $ zipWith (\y mu -> log y - mu) ys mus


-- lognormPDF behaves consistently with uniLognormPDF for univariate distributions
uniLognormPDF' y mu cov = (1 / sqrt (2*pi * cov)) * (1 / y) * exp ((-0.5) * (log y - mu)**2 / cov)


-- With 6 stdev, the Riemann sum is a slight overestimate. With 5 stdev, it's a slight underestimate
integralMaxStdev = 6 :: Double
integralNumDivisions = 12 :: Int
integralTrapezoidWidth = integralMaxStdev / fromIntegral integralNumDivisions


integralPoints :: [[Double]]
integralPoints = [[x, y, z] | x <- uniPoints, y <- uniPoints, z <- uniPoints, abs x + abs y + abs z < 15]
  where uniPoints = map ((* integralTrapezoidWidth) . fromIntegral) [(-integralNumDivisions)..integralNumDivisions]


integralRanges :: [[(Double, Double)]]
integralRanges = [[x, y, z] | x <- uniRanges, y <- uniRanges, z <- uniRanges]
  where uniPoints = map ((* integralTrapezoidWidth) . fromIntegral) [(-integralNumDivisions)..integralNumDivisions]
        uniRanges = zip uniPoints (tail uniPoints)


modelDistribution :: ([Double], [[Double]])
modelDistribution =
  let alphas = [0.08, 0.04, 0.03] :: [Double]
      sigmas = diagl [0.18, 0.18, 0.02] :: Matrix Double
      hedgeCorrelation = 0.5
      correlations = (3><3)
        [ 1, 0               , 0
        , 0, 1               , hedgeCorrelation
        , 0, hedgeCorrelation, 1
        ] :: Matrix Double
      covariances = toLists $ (sigmas <> correlations) <> sigmas
  in (alphas, covariances)


-- | Utility function with constant relative risk aversion.
crraUtility :: Double -> Double -> Double
crraUtility 1 money = log money
crraUtility rra money = (money**(1 - rra) - 1) / (1 - rra)


utility :: Double -> Double -> Double
utility wealth co2 = log wealth -- * co2


expectedUtility :: [Double] -> Double
expectedUtility weights' =
  sum $ map (
  \rectRanges ->
    let (alphas', covars') = modelDistribution
        weights = weights' ++ [1]  -- dummy weight for co2

        alphas = zipWith (*) weights alphas'
        covars = [[weights!!i * weights!!j * covars'!!i!!j | i <- [0..2]] | j <- [0..2]]

        mus = zipWith (\alpha row -> alpha - 0.5 * sum row) alphas covars
        stdevs = [sqrt $ covars!!i!!i | i <- [0..2]]
        transform point = zipWith4 (\mu sd wt x -> exp $ mu + sd * sqrt (abs wt) * signum wt * x) mus stdevs weights point

        y0 = transform $ map fst rectRanges
        y1 = transform $ map snd rectRanges
        width = product $ zipWith (-) y1 y0
        pdf0 = lognormPDF y0 mus covars
        pdf1 = lognormPDF y1 mus covars
        -- pdf = sqrt (pdf0 * pdf1)  -- geometric average (probably more accurate than arithmetic)
        pdf = (pdf0 + pdf1) / 2

        -- wealth = exp $ sum $ take 2 xs
        -- co2 = ys!!2
        -- u = utility wealth co2
    -- in u * pdf * width
    in pdf * width
  ) integralRanges


gradientDescentIO :: ([Double] -> Double) -> [Double] -> Double -> Int -> IO ([Double])
gradientDescentIO f initialGuess scale numSteps = go initialGuess numSteps
  where go :: [Double] -> Int -> IO ([Double])
        go x 0 = do
          printInfo x (f x)
          printf " (failed to converge after %d steps)\n" numSteps
          return x
        go x stepsLeft =
          let h = 0.0001
              fx = f x
              gradient = map (\i -> (f (updateAt i (+h) x) - fx) / h) [0..(length x - 1)]
              distance = sqrt $ sum $ map (**2) gradient
          in if isNaN distance || distance < 0.000001
             then do
               printInfo x fx
               printf " (converged after %d steps)\n" (numSteps - stepsLeft + 1)
               return x
             else do
               printInfo x fx
               printf " (step %d)\r" (numSteps - stepsLeft + 1)
               go (zipWith (+) x $ map (* scale) gradient) (stepsLeft - 1)

        printInfo x fx = printf "EU(%.4f, %.4f) = %.4f" (x!!0) (x!!1) fx


main :: IO ()
main = do
  let (alphas, covars) = modelDistribution

  print $ expectedUtility'' [0.5, 0.5]
  putStrLn ""
  -- res <- gradientDescentIO expectedUtility ([0.5, 0.5]) 10 100
  return ()
