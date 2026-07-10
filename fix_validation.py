import re

with open("Experts/v21.mq4", "r") as f:
    code = f.read()

# Chunk 1: Globals
code = code.replace(
    "double errGlobal = 999;\nint countTryed;",
    "double errGlobal = 999;\ndouble errValidation = 999;\ndouble bestErrValidation = 999999;\nint earlyStopCounter = 0;\nint countTryed;"
)

# Chunk 2: CalculateError to CalculateLoss
target_calc_err = """void CalculateError()
  {
   errGlobal = 0;
   int maxSample = trainSize - p - predictHorizon - trainBuffer;
   
   for(int iSample=0; iSample <= maxSample; iSample++)
     {
      double localMin = 999999;
      double localMax = -999999;
      for(int j=0; j<p; j++)
        {
         if(etalons[j+iSample] < localMin) localMin = etalons[j+iSample];
         if(etalons[j+iSample] > localMax) localMax = etalons[j+iSample];
        }
      double range = localMax - localMin;
      if(range == 0) range = 0.00001;
      
      double timeSin = timeSinArr[iSample + p - 1];
      double timeCos = timeCosArr[iSample + p - 1];
      double maFastNorm = maFastNormArr[iSample + p - 1];
      double maSlowNorm = maSlowNormArr[iSample + p - 1];
      double maFastSlope = maFastSlopeArr[iSample + p - 1];
      double maSlowSlope = maSlowSlopeArr[iSample + p - 1];

      for(int i=0; i<countHiddenNeuron; i++)
        {
         double wSum = 0;
         for(int j=0; j<p; j++)
           {
            double x = (etalons[j+iSample] - localMin) / range;
            wSum += weightsHidden[i][j] * x;
           }
         double rangeInput = (range / _Point) / 1000.0;
         wSum += weightsHidden[i][p] * rangeInput;
         
         double rsi14 = CalcRSI(etalons, iSample + p - 1, 14);
         double rsiInput14 = (rsi14 - 50.0) / 100.0;
         wSum += weightsHidden[i][p+1] * rsiInput14;
         
         double rsi7 = CalcRSI(etalons, iSample + p - 1, 7);
         double rsiInput7 = (rsi7 - 50.0) / 100.0;
         wSum += weightsHidden[i][p+2] * rsiInput7;
         
         wSum += weightsHidden[i][p+3] * timeSin;
         wSum += weightsHidden[i][p+4] * maFastNorm;
         wSum += weightsHidden[i][p+5] * maSlowNorm;
         wSum += weightsHidden[i][p+6] * maFastSlope;
         wSum += weightsHidden[i][p+7] * maSlowSlope;
         wSum += weightsHidden[i][p+8] * atrArr[iSample + p - 1];
         wSum += weightsHidden[i][p+9] * timeCos;
         
         wSum -= thresoldsHidden[i];
         weightedSums[i] = wSum;
         outputsHidden[i] = Sigmoid(wSum);
        }

      double outOutput[countClasses]; ArrayInitialize(outOutput, 0);
      for(int c=0; c<countClasses; c++)
        {
         double sumOut = 0;
         for(int i=0; i<countHiddenNeuron; i++) sumOut += weightsOutputLayer[c][i] * outputsHidden[i];
         sumOut -= thresoldOutputLayer[c];
         outOutput[c] = sumOut;
        }

      double maxOut = outOutput[0];
      for(int c=1; c<countClasses; c++) if(outOutput[c] > maxOut) maxOut = outOutput[c];
      double sumExp = 0;
      double probsTrain[countClasses]; ArrayInitialize(probsTrain, 0);
      for(int c=0; c<countClasses; c++) { probsTrain[c] = MathExp(outOutput[c] - maxOut); sumExp += probsTrain[c]; }
      for(int c=0; c<countClasses; c++) probsTrain[c] /= sumExp;
      
      double targetDeltaPoints = (rawPrices[iSample + p - 1 + predictHorizon] - rawPrices[iSample + p - 1]) / _Point;
      int targetClass = 3; // Flat
         if(targetDeltaPoints <= -classExtremeThreshold) targetClass = 0;
         else if(targetDeltaPoints <= -classStrongThreshold) targetClass = 1;
         else if(targetDeltaPoints <= -classWeakThreshold) targetClass = 2;
         else if(targetDeltaPoints >= classExtremeThreshold) targetClass = 6;
         else if(targetDeltaPoints >= classStrongThreshold) targetClass = 5;
         else if(targetDeltaPoints >= classWeakThreshold) targetClass = 4;
      
      double weightPenalty = (targetClass == 3) ? penaltyFlat : 1.0;
      errGlobal += (-MathLog(probsTrain[targetClass] + 1e-10)) * weightPenalty;
     }
   errGlobal = (errGlobal / (maxSample + 1));
  }"""

replacement_calc_err = """double CalculateLoss(int startIndex, int endIndex)
  {
   double err = 0;
   int count = 0;
   for(int iSample=startIndex; iSample <= endIndex; iSample++)
     {
      double localMin = 999999;
      double localMax = -999999;
      for(int j=0; j<p; j++)
        {
         if(etalons[j+iSample] < localMin) localMin = etalons[j+iSample];
         if(etalons[j+iSample] > localMax) localMax = etalons[j+iSample];
        }
      double range = localMax - localMin;
      if(range == 0) range = 0.00001;
      
      double timeSin = timeSinArr[iSample + p - 1];
      double timeCos = timeCosArr[iSample + p - 1];
      double maFastNorm = maFastNormArr[iSample + p - 1];
      double maSlowNorm = maSlowNormArr[iSample + p - 1];
      double maFastSlope = maFastSlopeArr[iSample + p - 1];
      double maSlowSlope = maSlowSlopeArr[iSample + p - 1];

      double outputsHiddenLocal[countHiddenNeuron];
      for(int i=0; i<countHiddenNeuron; i++)
        {
         double wSum = 0;
         for(int j=0; j<p; j++)
           {
            double x = (etalons[j+iSample] - localMin) / range;
            wSum += weightsHidden[i][j] * x;
           }
         double rangeInput = (range / _Point) / 1000.0;
         wSum += weightsHidden[i][p] * rangeInput;
         
         double rsi14 = CalcRSI(etalons, iSample + p - 1, 14);
         double rsiInput14 = (rsi14 - 50.0) / 100.0;
         wSum += weightsHidden[i][p+1] * rsiInput14;
         
         double rsi7 = CalcRSI(etalons, iSample + p - 1, 7);
         double rsiInput7 = (rsi7 - 50.0) / 100.0;
         wSum += weightsHidden[i][p+2] * rsiInput7;
         
         wSum += weightsHidden[i][p+3] * timeSin;
         wSum += weightsHidden[i][p+4] * maFastNorm;
         wSum += weightsHidden[i][p+5] * maSlowNorm;
         wSum += weightsHidden[i][p+6] * maFastSlope;
         wSum += weightsHidden[i][p+7] * maSlowSlope;
         wSum += weightsHidden[i][p+8] * atrArr[iSample + p - 1];
         wSum += weightsHidden[i][p+9] * timeCos;
         
         wSum -= thresoldsHidden[i];
         outputsHiddenLocal[i] = Sigmoid(wSum);
        }

      double outOutput[countClasses]; ArrayInitialize(outOutput, 0);
      for(int c=0; c<countClasses; c++)
        {
         double sumOut = 0;
         for(int i=0; i<countHiddenNeuron; i++) sumOut += weightsOutputLayer[c][i] * outputsHiddenLocal[i];
         sumOut -= thresoldOutputLayer[c];
         outOutput[c] = sumOut;
        }

      double maxOut = outOutput[0];
      for(int c=1; c<countClasses; c++) if(outOutput[c] > maxOut) maxOut = outOutput[c];
      double sumExp = 0;
      double probsTrain[countClasses]; ArrayInitialize(probsTrain, 0);
      for(int c=0; c<countClasses; c++) { probsTrain[c] = MathExp(outOutput[c] - maxOut); sumExp += probsTrain[c]; }
      for(int c=0; c<countClasses; c++) probsTrain[c] /= sumExp;
      
      double targetDeltaPoints = (rawPrices[iSample + p - 1 + predictHorizon] - rawPrices[iSample + p - 1]) / _Point;
      int targetClass = 3; // Flat
         if(targetDeltaPoints <= -classExtremeThreshold) targetClass = 0;
         else if(targetDeltaPoints <= -classStrongThreshold) targetClass = 1;
         else if(targetDeltaPoints <= -classWeakThreshold) targetClass = 2;
         else if(targetDeltaPoints >= classExtremeThreshold) targetClass = 6;
         else if(targetDeltaPoints >= classStrongThreshold) targetClass = 5;
         else if(targetDeltaPoints >= classWeakThreshold) targetClass = 4;
      
      double weightPenalty = (targetClass == 3) ? penaltyFlat : 1.0;
      err += (-MathLog(probsTrain[targetClass] + 1e-10)) * weightPenalty;
      count++;
     }
   if(count > 0) return err / count;
   return 0;
  }

void CalculateError()
  {
   int maxSample = trainSize - p - predictHorizon - trainBuffer;
   errGlobal = CalculateLoss(0, maxSample);
   
   if(trainBuffer > 0)
     {
      int valStart = maxSample + 1;
      int valEnd = trainSize - p - predictHorizon;
      if(valEnd >= valStart) errValidation = CalculateLoss(valStart, valEnd);
      else errValidation = 0;
     }
   else errValidation = 0;
  }"""

code = code.replace(target_calc_err, replacement_calc_err)

# Chunk 3: Train() Early Stopping
target_train_end = """        }
        
      errGlobal = (errGlobal / numSamples);

     }

   Print("Итерация " + countTeaches + " === Ошибко " + errGlobal + " Save Timer " + saveTimer);"""

replacement_train_end = """        }
        
      errGlobal = (errGlobal / numSamples);

      if(trainBuffer > 0)
        {
         int valStart = maxSample + 1;
         int valEnd = trainSize - p - predictHorizon;
         if(valEnd >= valStart)
           {
            errValidation = CalculateLoss(valStart, valEnd);
            if(errValidation > bestErrValidation)
              {
               earlyStopCounter++;
               if(earlyStopCounter >= 3)
                 {
                  nonTeacheble = true;
                  Print("Early Stopping: Обучение остановлено! Ошибка валидации растет.");
                  return;
                 }
              }
            else
              {
               bestErrValidation = errValidation;
               earlyStopCounter = 0;
              }
           }
        }

     }

   Print("Итерация " + countTeaches + " === Train Err: " + DoubleToString(errGlobal, 5) + " Val Err: " + DoubleToString(errValidation, 5));"""

code = code.replace(target_train_end, replacement_train_end)

# Chunk 4: Reset Early Stopping on New Bar
target_newbar = """   if(IsNewBar())
     {
      fastPassCounter++;
      countTryed = 0;
      nonTeacheble = false;
      ebobo = true;"""

replacement_newbar = """   if(IsNewBar())
     {
      fastPassCounter++;
      countTryed = 0;
      nonTeacheble = false;
      ebobo = true;
      bestErrValidation = 999999;
      earlyStopCounter = 0;"""

code = code.replace(target_newbar, replacement_newbar)

# Chunk 5: DrawLines UI
target_ui = """   lines[3] = "Mem: " + IntegerToString(trainSize) + " bars | Train buf: " + IntegerToString(trainBuffer) + " | +ATR +sin/cos(time)";
   lines[4] = "Global Error: " + DoubleToString(errGlobal, 5) + " / " + DoubleToString(tradeErrThresold, 5);
   lines[5] = "Passes Done: " + IntegerToString(countTeaches);"""

replacement_ui = """   lines[3] = "Mem: " + IntegerToString(trainSize) + " bars | Train buf: " + IntegerToString(trainBuffer) + " | +ATR +sin/cos(time)";
   lines[4] = "Train Err: " + DoubleToString(errGlobal, 5) + " | Val Err: " + DoubleToString(errValidation, 5);
   lines[5] = "Passes Done: " + IntegerToString(countTeaches);"""

code = code.replace(target_ui, replacement_ui)

with open("Experts/v21.mq4", "w") as f:
    f.write(code)

print("Validation and Early Stopping logic successfully implemented!")
