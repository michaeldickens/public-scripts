{- |
Module      : MissionHedging
Description :

Maintainer  : Michael Dickens <michael@mdickens.me>
Created     : 2022-02-21

A quantitative model of the value of mission hedging.

This program depends on these libraries: hmatrix, vector

To use this program, create a ModelParameters object defining the input
parameters. (See near the bottom of this file for some examples.) Then you can
pass it to one of these functions:

- gradientAscentIO: Find the asset allocation that maximizes utility.
- getGradient: Find the marginal utility of changes in asset allocation at a
  particluar point.
- expectedUtility: Calculate the expected utility of a particular asset allocation.

-}

module Main where

import Data.List as List
import qualified Data.Vector.Storable as Vec
import Debug.Trace
import Numeric.LinearAlgebra
import Prelude hiding ((<>))
import Text.Printf


{-| Data Definitions -}


-- | utility: Function from wealth and quantity of bad thing to utility
-- alphas: Arithmetic means of random variables
-- covariances: Covariances between random variables
-- dimension: Number of random variables
data ModelParameters = ModelParameters
  { utility :: Double -> Double -> Double
  , alphas :: [Double]
  , covariances :: [[Double]]
  , dimension :: Int
  , predicted :: [Double]
  }


{-|

Helper Functions

-}


traceShowSelf x = traceShow x x


-- | Update the element of a list at index `i` by passing it to the given
-- function.
updateAt :: Int -> (a -> a) -> [a] -> [a]
updateAt i f xs = go i xs
  where go 0 (x:xs) = (f x) : xs
        go j (x:xs) = x : (go (j-1) xs)


{-|

Probability Distribution Functions

-}


-- | Probability density of a standard normal distribution.
standardNormalPDF :: [Double] -> Double
standardNormalPDF zs = (1 / sqrt ((2 * pi)**(fromIntegral $ length zs))) * exp (-0.5 * (sum $ map (**2) zs))


-- | Probability density of a multivariate normal distribution, parameterized by
-- a mean vector and a covariance matrix.
multiNormPDF :: [Double] -> [Double] -> [[Double]] -> Double
multiNormPDF xs means covars =
  (exp $ (-0.5) * ((residuals <# inv covMat) `dot` residuals))
  /
  (sqrt $ (2 * pi)**k * det covMat)
  where residuals = Vec.fromList $ zipWith (-) xs means
        k = fromIntegral $ length xs
        covMat = fromLists covars


-- | Multivariate lognormal PDF.
--
-- Same as a normal PDF but where input vector `ys` is transformed by taking the
-- logarithm, and whole thing is multiplied by `det(Jacobian(ys))` = `1 /
-- product ys` to adjust the base area of the Riemann sum.
-- https://stats.stackexchange.com/questions/214997/multivariate-log-normal-probabiltiy-density-function-pdf
-- https://math.stackexchange.com/questions/267267/intuitive-proof-of-multivariable-changing-of-variables-formula-jacobian-withou
--
-- ys: n-dimensional random variable
--
-- mus: means of the underlying normal distributions
--
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


{-| Expected Utility Calculation Functions -}


-- | Internal parameter. When integrating over a normal probability
-- distribution, cover this many standard deviations in each direction.
--
-- With 6 stdev, the integral is a slight overestimate. With 5 stdev, it's a
-- slight underestimate.
--
-- For a single variable, 5 standard deviations cover 99.99994% of the
-- probability space (less than one millionth of the space is excluded). For
-- independent three-variable, 5 stdevs cover 99.9998%, and the coverage goes up
-- as correlation increases. (That's better coverage than Amazon S3)
integralMaxStdev = 6 :: Double


-- | A list of ranges to be used for numeric integration. A range is defined by
-- a list of pairs giving the minimum and maximum of the range, with one pair
-- for each dimension.
--
-- dimension: Number of dimensions over which to integrate.
--
-- numTrapezoids: Number of trapezoids to use in the positive direction along
-- one dimension. The negative direction uses the same number of trapezoids. The
-- total number of trapezoids will be `(2*numTrapezoids)^dimension`.
integralRanges :: Int -> Int -> [[(Double, Double)]]
integralRanges dimension numTrapezoids =
  -- For each iteration, take the existing list of ranges, copy each list `len
  -- uniRanges` times, and prepend one element of `uniRanges` to each copy
  (iterate (concatMap (\xs -> map (:xs) uniRanges)) [[]]) !! dimension

  where trapezoidWidth = integralMaxStdev / fromIntegral numTrapezoids
        uniCoords = map ((* trapezoidWidth) . fromIntegral) [(-numTrapezoids)..numTrapezoids]
        uniRanges = zip uniCoords (tail uniCoords) :: [(Double, Double)]


-- | Numerically compute expected utility using the Trapezoid Rule.
--
-- numTrapezoids: Number of trapezoids to use per direction. For a
-- 3-dimensional integral, there are 6 directions = (2*numTrapezoids)^3 total
-- trapezoids.
--
-- weights': a length-2 vector of weights for the first two variables (market
-- asset and hedge asset). The third variable (quantity of bad thing) always has
-- a weight of 1.
expectedUtilityTrapezoid :: ModelParameters -> Int -> [Double] -> Double
expectedUtilityTrapezoid (ModelParameters utility alphas' covars' dim _) numTrapezoids weights' =
  let weights = weights' ++ [1]  -- dummy weight for badThing

      -- Note: If numYears > 1, gradientAscent needs to use a smaller `scale` or
      -- else it will fail to converge.
      numYears = 1

      -- TODO: Not 100% sure I am scaling correctly with numYears
      alphas = map (* numYears) $ zipWith (*) weights alphas'
      covars = [[numYears * weights!!i * weights!!j * covars'!!i!!j | i <- [0..dim-1]] | j <- [0..dim-1]]

      stdevs = [sqrt $ covars!!i!!i | i <- [0..dim-1]]
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

            -- Recall that ys are already weighted by portfolio allocation. You
            -- can think of this as holding y[0] for a proportionate length of
            -- time, then holding y[1].
            wealth = product $ init ys

            badThing = last ys
            u = utility wealth badThing
        in u * pdf * width
      ) $ integralRanges dim numTrapezoids


-- | Numerically compute expected utility of a portfolio with the given weights.
--
-- Inputs should satisfy `length weights = dimension modelParams - 1` because
-- the last dimension covers the bad thing, which does not have a portfolio
-- weight.
--
-- Implemented using Richardson's extrapolation formula for the trapezoid rule.
-- Empirical tests suggest that, for n = number of trapezoids, Richardson(2n=8)
-- is more accurate than Trapezoid(n=40), and >100x faster.
--
-- See https://mathforcollege.com/nm/mws/gen/07int/mws_gen_int_txt_romberg.pdf
expectedUtility :: ModelParameters -> [Double] -> Double
expectedUtility modelParams weights =
  let numTrapezoids = 6
      bigTrap = expectedUtilityTrapezoid modelParams numTrapezoids weights
      smallTrap = expectedUtilityTrapezoid modelParams (2 * numTrapezoids) weights
  in smallTrap + (smallTrap - bigTrap) / 3


{-|

Optimization Functions

-}


-- | Numerically compute the gradient of a multivariate function.
getGradient :: ([Double] -> Double) -> [Double] -> [Double]
getGradient f x = getGradient' f x (f x)


-- | Numerically compute the gradient of a multivariate function, supplying the function value at the current location as input. Use this rather than `getGradient` when computing `f x` is expensive.
getGradient' :: ([Double] -> Double) -> [Double] -> Double -> [Double]
getGradient' f x fx = map (\i -> (f (updateAt i (+h) x) - fx) / h) [0..(length x - 1)]
  where h = 0.0001



-- | Perform gradient ascent, printing the results at each step.
--
-- f :: [Double] -> Double: The function to be maximized.
--
-- initialGuess :: [Double]: The point at which to begin gradient ascent.
--
-- scale :: Double: A parameter that determines the speed of ascent for each
-- parameter. A larger value means it will converge faster, but too large of a
-- scale means it will fail to converge.
--
-- Gradient ascent halts if it does not converge after a certain number of steps.
--
-- Example usage:
--     gradientDescentIO (expectedUtility standardParams) [1, 1] 10
gradientAscentIO :: ([Double] -> Double) -> [Double] -> Double -> IO ([Double])
gradientAscentIO f initialGuess scale = go initialGuess numSteps
  where numSteps = 100
        go :: [Double] -> Int -> IO ([Double])
        go x 0 = do
          printInfo x (f x)
          printf " (failed to converge after %d steps)\n" numSteps
          return x
        go x stepsLeft =
          let fx = f x
              gradient = getGradient' f x fx
              distance = sqrt $ sum $ map (**2) gradient
          in if isNaN distance || distance < 0.00001
             then do
               printInfo x fx
               printf " (converged after %d steps)\n" (numSteps - stepsLeft + 1)
               return x
             else do
               printInfo x fx
               printf " (step %d)\r" (numSteps - stepsLeft + 1)
               go (zipWith (\xi g -> xi + scale * g) x gradient) (stepsLeft - 1)

        printInfo x fx = printf "%s\rEU(%.3f, %.3f) = %.4f" (take 40 $ repeat ' ') (x!!0) (x!!1) fx


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


{-|

Utility Functions

-}


-- | Utility function with constant relative risk aversion of wealth and linear
-- scaling with the bad thing.
--
-- For rra > 1, this utility function has the following properties:
--
-- 1. dU/dw > 0                  (utility increases with wealth)
-- 2. dU/db < 0                  (utility decreases with bad thing)
-- 3. d^2 U / dw db > 0          (wealth becomes more useful as bad thing increases)
-- 4. d^2 U / dw^2 < 0           (diminishing marginal utility of wealth)
-- 5. constant relative risk aversion with respect to wealth
-- 6. lim(w -> inf) dU/db = 0    (infinite wealth fully cancels out bad thing)
crraUtility :: Double -> Double -> Double -> Double
crraUtility 1 wealth bad = bad * log wealth - bad
crraUtility rra wealth bad = bad * (wealth**(1 - rra)) / (1 - rra)


-- | Assumptions behind this utility function:
-- - AGI arrives by some date -> equivalently, AGI progresses at some rate
-- - Doing quantity Q of research has a probability p of preventing extinction
-- - Need enough money to do Q research -> equivalently, need some growth rate
-- - Goal: maximize probability of preventing extinction
--
-- Therefore, if growth rate of wealth exceeds a constant multiple of the growth
-- rate of AI progress, utility = 1. Otherwise, utility = 0. Use a logistic
-- function to make it smooth.
--
-- Referring to the properties of `crraUtility`, this function has properties 1
-- and 2, but not 3, 4, 5, or 6. Generally speaking, this utility function is
-- less well-behaved and harder to reason about.
--
-- Gradient ascent over expected logistic utility fails to converge.
--
-- growthScale: The relative scale of the growth rates of wealth and the bad
-- thing. Higher scale = the value of wealth grows faster (i.e., less wealth is
-- required).
logisticUtility :: Double -> Double -> Double -> Double
logisticUtility growthScale wealth bad = 1 / (1 + exp (1 - growthScale * log wealth / log bad))


-- | Utility function where the interaction between `wealth` and `bad` behaves
-- like `crraUtility', but wealth also (independently) increases utility
-- logarithmically.
--
-- This function satisfies all six criteria except for constant relative risk
-- aversion.
unboundedUtility :: Double -> Double -> Double -> Double
unboundedUtility rra wealth bad = log wealth + crraUtility rra wealth bad


{-|

Main

-}


-- | Parameter set with three variables:
-- 1. mean-variance optimal asset
-- 2. hedge asset
-- 3. bad thing
standardParams :: ModelParameters
standardParams =
  let alphas = [0.06, 0.00, 0.07] :: [Double]
      sigmas' = [0.13, 0.22, 0.21]
      sigmas = diagl sigmas' :: Matrix Double
      hedgeCorr = 0.72 -- correlation between hedge and bad thing
      correlations = (3><3)
        [ 1, 0        , 0
        , 0, 1        , hedgeCorr
        , 0, hedgeCorr, 1
        ] :: Matrix Double
      covariances = toLists $ (sigmas <> correlations) <> sigmas
      rra = 1.1
      utility = crraUtility rra
      predicted = [ head alphas / (head sigmas' ** 2 * rra)
                  , hedgeCorr * (sigmas'!!2) / (sigmas'!!1 * rra)
                  ]
  in ModelParameters utility alphas covariances 3 predicted


-- | Parameter set with four variables:
-- 1. mean-variance optimal asset
-- 2. undesirable legacy asset
-- 3. hedge asset
-- 4. bad thing
--
-- Note: Computing expected utility over a four-dimensional space is very slow.
legacyParams :: ModelParameters
legacyParams =
  let alphas = [0.08, 0.08, 0.00, 0.10] :: [Double]
      sigmas = diagl [0.18, 0.36, 0.18, 0.30] :: Matrix Double
      hedgeCorr = 0.5    -- correlation between hedge and bad thing
      legacyCorr = 0.5   -- correlation between MVO asset and legacy asset
      correlations = (4><4)
        [ 1         , legacyCorr, 0        , 0
        , legacyCorr, 1         , 0        , 0
        , 0         , 0         , 1        , hedgeCorr
        , 0         , 0         , hedgeCorr, 1
        ] :: Matrix Double
      covariances = toLists $ (sigmas <> correlations) <> sigmas
      utility = crraUtility 1.5
  in ModelParameters utility alphas covariances 4 []


main :: IO ()
main = do
  -- print $ getGradient (expectedUtility legacyParams) [0.5, 0.5, 0.001]
  printf "predicted allocation: %.3f, %.3f\n" (predicted standardParams!!0) (predicted standardParams!!1)
  weights <- gradientAscentIO (expectedUtility standardParams) [1.5, 0.25] 15
  return ()
