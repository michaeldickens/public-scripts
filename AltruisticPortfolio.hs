
expectedUtility :: Double -> Double -> Double -> Double -> Double
expectedUtility lev m1 s1 r =
  let w0 = 0.999
      w1 = 0.001
      m0 = 0.02
      s0 = 0.16
      a0 = m0 + s0**2/2
      a1 = m1 + s1**2/2
      a = (1 + w0*a0 + lev*w1*a1)
  in log a - ((w0*s0)**2 + (lev*w1*s1)**2 + w0*lev*w1*r*s0*s1) / (2 * a**2)


main :: IO ()
main = do
  putStrLn $ "VMOT: " ++ show (expectedUtility (16/13) 0.06 0.13 (0.5))
  putStrLn $ "market neutral 1: " ++ show (expectedUtility (16/8) 0.03 0.08 (-0.2))
  putStrLn $ "market neutral 2: " ++ show (expectedUtility (16/7) 0.03 0.07 (-0.2))
  putStrLn $ "market neutral 3: " ++ show (expectedUtility (16/6) 0.03 0.06 (-0.2))
