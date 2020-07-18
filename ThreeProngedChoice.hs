{- |

ThreeProngedChoice.hs

Created 2020-07-02

-}

module ThreeProngedChoice where

import Data.Function (on)
import Data.List
import Text.Printf

utilityFunction :: Double -> Double -> Double
utilityFunction eta consumption
  | eta == 1 = log consumption
  | otherwise = (consumption**(1 - eta) + 1) / (1 - eta)


totalUtility :: Double -> Double -> Double -> Double -> Double -> (Double, Double) -> Double
totalUtility alpha delta r eta numPeriods (propDiscount, propConsumption) = go 0 100 1 1 0
  where go :: Double -> Double -> Double -> Double -> Double -> Double
        go period capital spendingOnDiscount discountFactor utility
          | period == numPeriods = utility
          | otherwise =
            go (period + 1)
            (capital * (1 - propDiscount - propConsumption) * (1 + r))
            (spendingOnDiscount + capital * propDiscount)
            (discountFactor * (1 - delta * spendingOnDiscount**(-alpha)))
            (utility + discountFactor * utilityFunction eta (1 + capital * propConsumption))


greedyOptimize :: ((Double, Double) -> Double) -> ((Double, Double) -> Bool) -> (Double, Double) ->
                  Double -> (Double, Double)
greedyOptimize f constraint guess depth =
  let variants = filter constraint
                 $ [guess, (0.99 * fst guess, snd guess), (1.01 * fst guess, snd guess),
                  (fst guess, 0.99 * snd guess), (fst guess, 1.01 * snd guess),
                  (1.01 * fst guess, 0.99 * snd guess), (0.99 * fst guess, 1.01 * snd guess),
                  (0.99 * fst guess, 0.99 * snd guess), (1.01 * fst guess, 1.01 * snd guess)]
      bestVariant = fst $ maximumBy (compare `on` snd) $ zip variants (map f variants)
  in if bestVariant == guess
     then guess
     else if depth >= 1000
          then bestVariant
          else greedyOptimize f constraint bestVariant (depth + 1)


main :: IO ()
main = do
  let f = totalUtility 1 0.02 0.05 1 100
  let opt = greedyOptimize f ((\(x, y) -> x + y < 1)) (0.01, 0.3) 0
  printf "x(t) = %.4f%%, c(t) = %.4f%%\n" (fst opt * 100) (snd opt * 100)
  print $ f opt
