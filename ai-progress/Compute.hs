{- |
Module      : Compute
Description : Compute trends using the database from Jaime Sevilla et al.

Maintainer  : Michael Dickens <michael@mdickens.me>
Created     : 2022-03-12

-}

module Compute where

import Data.Function (on)
import Data.List
import Data.List.Split
import Text.Printf

data Metric = Compute | NumParams deriving (Eq)

len :: [Double] -> Double
len = fromIntegral . length

mean :: [Double] -> Double
mean xs = sum xs / len xs

stdDev :: [Double] -> Double
stdDev xs =
  let mu = mean xs
  in sqrt $ (1 / (len xs - 1)) * (sum $ map (\x -> (x - mu)**2) xs)

main :: IO ()
main = do
  let metric = NumParams
  let transform = (** (0.35))
  -- let transform = (** (-0.050))      -- for compute
  -- let transform = (** (-0.076))   -- for num params
  -- let transform = id
  -- let transform = log

  file <- readFile "data/compute.csv"
  let rows = map (splitOn ",") $ tail $ lines file
  let rowsByYear = groupBy ((==) `on` head) rows
  let index = if metric == Compute then 2 else 1
  let compute' = map (filter (> 0)
                      . map (\row -> read (row!!index) :: Double)
                      . filter (\row -> length (row!!index) > 0))
                 $ filter (\row -> let yr = ((read $ head $ head row) :: Int)
                                   in yr >= 2012 && yr <= 2021) rowsByYear

  let compute = map (map transform) compute'
  let avgCompute = map mean compute
  let growth = zipWith (\x y -> log $ y/x) avgCompute (tail avgCompute)
  -- let growth = zipWith (-) avgCompute (tail avgCompute)
  putStrLn $ intercalate ", " $ map (printf "%.2f" . (*100)) growth
  printf "mean %.1f%%, stdev %.1f%%\n" (100 * mean growth) (100 * stdDev growth)

  -- A naive attempt to find the 'true' volatility
  -- let stderrs = map (\xs -> stdDev xs / len xs / mean xs) compute
  -- let aggStderr = (sum $ zipWith (*) stderrs (map len compute)) / (sum $ map len compute)
  -- print aggStderr
  -- putStrLn $ "adjusted stdev: " ++ show (stdDev growth - aggStderr)
