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


-- | Multivariate lognormal PDF
--
-- Same as a normal PDF but where input vector `ys` is transformed by taking the
-- logarithm, and whole thing is multiplied by `det(Jacobian(ys))` = `1 /
-- product ys` to adjust the base area of the Riemann sum.
-- https://stats.stackexchange.com/questions/214997/multivariate-log-normal-probabiltiy-density-function-pdf
-- https://math.stackexchange.com/questions/267267/intuitive-proof-of-multivariable-changing-of-variables-formula-jacobian-withou
--
-- ys: n-dimensional random variable
-- mus: means of the underlying normal distributions
-- covars: covariance matrix for the underlying normal distributions
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
integralMaxStdev = 5 :: Double
integralNumDivisions = 15 :: Int
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
  let alphas = [0.08, 0.00, 0.05] :: [Double]
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
crraUtility 1 wealth = log wealth
crraUtility rra wealth = (wealth**(1 - rra) - 1) / (1 - rra)


-- | For rra > 1, this utility function has the following properties:
--
-- dU/dw > 0                  (utility increases with wealth)
-- dU/db < 0                  (utility decreases with bad thing)
-- d^2 U / dw^2 < 0           (diminishing marginal utility of wealth)
-- d^2 U / dw db > 0          (wealth becomes more useful as bad thing increases)
-- lim(w -> inf) U(w, b) = 0  (infinite wealth fully cancels out bad thing)
multiCrraUtility :: Double -> Double -> Double -> Double
multiCrraUtility rra wealth bad = bad * (wealth**(1 - rra)) / (1 - rra)


ostensiblyIrrelevantOffset = 0


-- TODO: optimal allocation changes wrt k, but it shouldn't
utility :: Double -> Double -> Double
utility = multiCrraUtility rra
-- utility wealth bad = (wealth**(1 - rra) - ostensiblyIrrelevantOffset) / (1 - rra)
-- utility wealth bad = log wealth * bad
  where rra = 1.5


expectedUtility :: [Double] -> Double
expectedUtility weights' =
  let (alphas', covars') = modelDistribution
      weights = weights' ++ [1]  -- dummy weight for badThing

      alphas = zipWith (*) weights alphas'
      covars = [[weights!!i * weights!!j * covars'!!i!!j | i <- [0..2]] | j <- [0..2]]

      stdevs = [sqrt $ covars!!i!!i | i <- [0..2]]
      mus = zipWith (\alpha sd -> alpha - 0.5 * sd**2) alphas stdevs

      -- | Transform from standard normal space to nonstandard lognormal space
      transform point = zipWith3 (\mu sd x -> exp $ mu + sd * x) mus stdevs point
  in sum $ map (
      \rectRanges ->
        let yLower = transform $ map fst rectRanges
            yUpper = transform $ map snd rectRanges
            ys = transform $ map (\(e, s) -> (e + s) / 2) rectRanges
            width = abs $ product $ zipWith (-) yUpper yLower
            pdf = lognormPDF ys mus covars

            -- Recall that ys are already weighted by portfolio allocation. Per
            -- Irlam, you can think of this as holding y[0] for a proportionate
            -- length of time, then holding y[1].
            wealth = product $ take 2 ys

            badThing = ys!!2
            u = utility wealth badThing
        in u * pdf * width
      ) integralRanges


getGradient :: ([Double] -> Double) -> [Double] -> [Double]
getGradient f x = map (\i -> (f (updateAt i (+h) x) - fx) / h) [0..(length x - 1)]
  where fx = f x
        h = 0.0001


-- | Perform gradient ascent, printing the results at each step.
--
-- - f :: [Double] -> Double: The function to be maximized.
-- - initialGuess :: [Double]: The point at which to begin gradient ascent.
-- - scale :: [Double]: A parameter that determines the speed of ascent for each
--   parameter. A larger value means it will converge faster, but too large of a
--   scale means it will fail to converge.
--
-- Gradient ascent halts if it does not converge after a certain number of steps.
gradientAscentIO :: ([Double] -> Double) -> [Double] -> [Double] -> IO ([Double])
gradientAscentIO f initialGuess scale = go initialGuess numSteps
  where numSteps = 100
        go :: [Double] -> Int -> IO ([Double])
        go x 0 = do
          printInfo x (f x)
          printf " (failed to converge after %d steps)\n" numSteps
          return x
        go x stepsLeft =
          let h = 0.0001
              fx = f x
              gradient = map (\i -> (f (updateAt i (+h) x) - fx) / h) [0..(length x - 1)]
              distance = sqrt $ sum $ map (**2) gradient
          in if isNaN distance || distance < 0.00001
             then do
               printInfo x fx
               printf " (converged after %d steps)\n" (numSteps - stepsLeft + 1)
               return x
             else do
               printInfo x fx
               printf " (step %d)\r" (numSteps - stepsLeft + 1)
               go (zipWith3 (\xi s g -> xi + s * g) x scale gradient) (stepsLeft - 1)

        printInfo x fx = printf "EU(%.3f, %.3f) = %.4f" (x!!0) (x!!1) fx


-- | Approximate geometric mean of an investment portfolio.
--
-- From Estrada (2010), "Geometric Mean Maximization: An Overlooked Portfolio Approach?"
-- https://blog.iese.edu/jestrada/files/2012/06/GMM-Extended.pdf
--
-- TODO: For some reason, the naive (alpha - cov/2) formula is a much better
-- approximation, especially close to the GMM-maximizing weights.
geometricMean :: [Double] -> [[Double]] -> [Double] -> Double
geometricMean alphas covars weights =
  -- log (1 + weightVec `dot` alphaVec)
  -- - 0.5 * ((weightVec <# covMat) `dot` weightVec) / (1 + weightVec `dot` alphaVec)**2
  weightVec `dot` alphaVec - 0.5 * ((weightVec <# covMat) `dot` weightVec)
  where alphaVec = Vec.fromList $ take 2 alphas
        covMat = takeRows 2 $ takeColumns 2 $ fromLists covars
        varVec = takeDiag covMat
        weightVec = Vec.fromList $ take 2 weights
        muVec = Vec.zipWith (\a v -> a - v**2 / 2) alphaVec varVec


-- TODO: The problem is that integration isn't very accurate when weights are small, eg with hedge weight = 0.01, integral only covers 45% of the probability space
main :: IO ()
main = do
  let (alphas, covars) = modelDistribution

  let printEU x = printf "EU(%.3f, %.3f) = %.4f\n" (x!!0) (x!!1) (expectedUtility x)
  -- print $ (10 - 10*ostensiblyIrrelevantOffset) + expectedUtility [1.878, -0.012]
  -- print $ getGradient expectedUtility [1.878, -0.012]
  weights <- gradientAscentIO expectedUtility ([1.5, 0.25]) [15, 15]
  -- sequence $ map (\hedge -> printf "%.4f: %.5f\n" (hedge::Double) (expectedUtility [1.557, hedge])) $ filter (/= 0) $ map ((/ 10000) . fromIntegral) [(-50 :: Int)..50]
  -- weights <- gradientAscentIO expectedUtility ([0.5, -1]) [5, 5]
  return ()
