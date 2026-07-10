//+------------------------------------------------------------------+
//|                                                            v21.mq4 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "2.10"
#property strict

#define p 36
#define countHiddenNeuron 10
#define countClasses 7
#define trainSize 1536

input double tpMultiplier = 1.0;
input bool readInitData = false;
input int takeProfit = 300;
input int stopLoss = 400;
input int predictHorizon = 8;
input int predictShift = 2;
input int classExtremeThreshold = 500;
input int classStrongThreshold = 200; // Порог сильного движения (пункты)
input int classWeakThreshold = 50;    // Порог слабого движения (пункты)
input double probTradeThreshold = 0.6; // Мин. вероятность (0.0 - 1.0)
input double penaltyFlat = 0.1;       // Штраф за флэт

input int pass = 1;
int maxTeach = 1000;
input double wRange = 0.35; // диапозон случайных значений весов
input bool saveLearningProgress;
input bool ovverideSaveFile;
input int saveTimerThresold = 100;
input bool allowSkipTrain = false;
input int skipTrainMaxPasses = 24;
input double aStep = 0.05;
input bool onlyOneOrder = true;
input int magicNumber = 123456;
input double tradeErrThresold = 0.110;
input int trainBuffer = 136; // Кол-во свежих баров, которые НЕ используются при обучении (тестовый буфер)

input bool useTrailingStop = false; // Включить трейлинг-стоп?
input int trailingStart = 300;      // При какой прибыли в пунктах включать трейлинг
input int trailingStep = 50;        // Шаг подтягивания стопа (пункты)

input bool useFixedSeed = true; // Использовать фиксированное зерно?
input int randomSeed = 1;    // Зерно для генератора весов (мозга)
input int fontSize = 14;     // Размер шрифта инфопанели
input int lineSpacing = 25;  // Межстрочный интервал инфопанели
input color infoTextColor = clrNONE; // Цвет текста (clrNONE = авто)

// p+10 входов: 36 дельт + Range + RSI(14) + RSI(7) + sin(time) + MA_Fast_Norm + MA_Slow_Norm + MA_Fast_Slope + MA_Slow_Slope + ATR(14) + cos(time)
double weightsHidden[countHiddenNeuron][p+10]; 
double thresoldsHidden[countHiddenNeuron];
double weightedSums[countHiddenNeuron];
double outputsHidden[countHiddenNeuron];
double weightsOutputLayer[countClasses][countHiddenNeuron];
double thresoldOutputLayer[countClasses];
double predictedProbs[countClasses];
double yValues[trainSize];
double etalons[trainSize];
double rawPrices[trainSize];
double maFastNormArr[trainSize];
double maSlowNormArr[trainSize];
double maFastSlopeArr[trainSize];
double maSlowSlopeArr[trainSize];
double atrArr[trainSize];
double timeSinArr[trainSize];
double timeCosArr[trainSize];

double predictedDelta;

// --- Adam Optimizer variables ---
double beta1 = 0.9;
double beta2 = 0.999;
double epsilon = 1e-8;
long adam_t = 0;

double m_weightsHidden[countHiddenNeuron][p+10];
double v_weightsHidden[countHiddenNeuron][p+10];
double m_thresoldsHidden[countHiddenNeuron];
double v_thresoldsHidden[countHiddenNeuron];
double m_weightsOutputLayer[countClasses][countHiddenNeuron];
double v_weightsOutputLayer[countClasses][countHiddenNeuron];
double m_thresoldOutputLayer[countClasses];
double v_thresoldOutputLayer[countClasses];
// --------------------------------

double offsetY = 0;
int countTeaches = 0;

int saveTimer = 0;
int ticket;

bool ebobo;
bool nonTeacheble;
double errGlobal = 999;
double errValidation = 999;
double bestErrValidation = 999999;
int earlyStopCounter = 0;
double teachErrThresold = 3.5;
int fastPassCounter;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(useFixedSeed) MathSrand(randomSeed);
   CleanupGraphics();
   InitBars();
   InitWeights();
   return(INIT_SUCCEEDED);
  }

void CleanupGraphics()
  {
   ObjectsDeleteAll(0, "Ebat_");
   ObjectsDeleteAll(0, "NN_");
   ObjectsDeleteAll(0, "PredictLine");
   ObjectsDeleteAll(0, "InfoLabel_");
  }

//+------------------------------------------------------------------+
//| File I/O & Weights Setup                                         |
//+------------------------------------------------------------------+
void InitWeights()
  {
   countTeaches = 0;
   errGlobal = 999;
   if(readInitData)
     {
      string strWeightsHidden;
      int handle = FileOpen("Exp20_weightsHidden_" + IntegerToString(magicNumber) + "_" + IntegerToString(0) + "_" + DoubleToString(tpMultiplier, 1) + "_" + IntegerToString(p+9) + "_" + IntegerToString(countHiddenNeuron) + ".txt", FILE_TXT|FILE_READ);
      if(handle > 0)
        {
         strWeightsHidden = FileReadString(handle);
         string result[];
         int k = StringSplit(strWeightsHidden, '|', result) - 1;
         if(k/2 != countHiddenNeuron * (p+10))
           {
            Print("!!!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!!");
           }
         else
           {
            int idxRead = 0;
            for(int i=0; i < countHiddenNeuron; i++)
               for(int j=0; j <= p+9; j++)
                 {
                  weightsHidden[i][j] = StringToDouble(result[idxRead*2]);
                  idxRead++;
                 }
           }
         FileClose(handle);
        }
      else
        {
         Print("Ebala Syet ************ " + GetLastError());
         
         // FALLBACK FIX: IF FILE NOT FOUND BUT READINITDATA TRUE
         string opta = "";
         for(int i=0; i<countHiddenNeuron; i++)
           {
            for(int j=0; j<=p+9; j++) weightsHidden[i][j] = RandomDouble(-wRange, wRange);
            thresoldsHidden[i] = RandomDouble(-wRange, wRange);
           }
         for(int c=0; c<countClasses; c++)
           {
            for(int i=0; i<countHiddenNeuron; i++) weightsOutputLayer[c][i] = RandomDouble(-0.5, 0.5);
            thresoldOutputLayer[c] = RandomDouble(-0.5, 0.5);
           }
        }
      
      string strThresoldsHidden;
      handle = FileOpen("Exp20_thresoldsHidden_" + IntegerToString(magicNumber) + "_" + IntegerToString(0) + "_" + DoubleToString(tpMultiplier, 1) + "_" + IntegerToString(p) + "_" + IntegerToString(countHiddenNeuron) + ".txt", FILE_TXT|FILE_READ);
      if(handle > 0)
        {
         strThresoldsHidden = FileReadString(handle);
         string result[];
         int k = StringSplit(strThresoldsHidden, '|', result) - 1;
         if(k/2 != countHiddenNeuron)
           {
            Print("!!!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!!");
           }
         else
           {
            int idxRead = 0;
            for(int i=0; i<countHiddenNeuron; i++)
              {
               thresoldsHidden[i] = StringToDouble(result[idxRead*2]);
               idxRead++;
              }
           }
         FileClose(handle);
        }
      
      LoadWeightsOutputLayer(false);
      
      ArrayInitialize(m_weightsHidden, 0.0);
      ArrayInitialize(v_weightsHidden, 0.0);
      ArrayInitialize(m_thresoldsHidden, 0.0);
      ArrayInitialize(v_thresoldsHidden, 0.0);
      ArrayInitialize(m_weightsOutputLayer, 0.0);
      ArrayInitialize(v_weightsOutputLayer, 0.0);
      ArrayInitialize(m_thresoldOutputLayer, 0.0);
      ArrayInitialize(v_thresoldOutputLayer, 0.0);
      adam_t = 0;
      Print("Successfully loaded weights from files!");
     }
   else
     {
      string opta = "";
      for(int i=0; i<countHiddenNeuron; i++)
        {
         for(int j=0; j<=p+9; j++)
           {
            weightsHidden[i][j] = RandomDouble(-wRange, wRange);
            opta += weightsHidden[i][j] + "|=|";
           }
         thresoldsHidden[i] = RandomDouble(-wRange, wRange);
        }
      for(int c=0; c<countClasses; c++)
        {
         for(int i=0; i<countHiddenNeuron; i++) weightsOutputLayer[c][i] = RandomDouble(-0.5, 0.5);
         thresoldOutputLayer[c] = RandomDouble(-0.5, 0.5);
        }
      WriteToFile(opta, "Exp20_weightsHidden_" + IntegerToString(magicNumber) + "_" + IntegerToString(0) + "_" + DoubleToString(tpMultiplier, 1) + "_" + IntegerToString(p+9) + "_" + IntegerToString(countHiddenNeuron) + ".txt");
   
      SaveThresoldsHidden(false);
      SaveWeightsOutputLayer(false);
      
      ArrayInitialize(m_weightsHidden, 0.0);
      ArrayInitialize(v_weightsHidden, 0.0);
      ArrayInitialize(m_thresoldsHidden, 0.0);
      ArrayInitialize(v_thresoldsHidden, 0.0);
      ArrayInitialize(m_weightsOutputLayer, 0.0);
      ArrayInitialize(v_weightsOutputLayer, 0.0);
      ArrayInitialize(m_thresoldOutputLayer, 0.0);
      ArrayInitialize(v_thresoldOutputLayer, 0.0);
      adam_t = 0;
      Print("======================================= Веса Инициализированы ===================================");
     }
  }
  
void SaveWeightsData(bool isLearning)
  {
   SaveWeightsHiddenLayer(isLearning);
   SaveThresoldsHidden(isLearning);
   SaveWeightsOutputLayer(isLearning);
  }

void SaveWeightsHiddenLayer(bool isLearning)
  {
   string data = "";
   for(int i=0; i<countHiddenNeuron; i++)
      for(int j=0; j<=p+9; j++)
         data += weightsHidden[i][j] + "|=|";
   if(isLearning) WriteToFile(data, "L_Exp20_weightsHidden_" + IntegerToString(magicNumber) + "_" + IntegerToString(0) + "_" + DoubleToString(tpMultiplier, 1) + "_" + IntegerToString(p+9) + "_" + IntegerToString(countHiddenNeuron) + ".txt");
   else WriteToFile(data, "Exp20_weightsHidden_" + IntegerToString(magicNumber) + "_" + IntegerToString(0) + "_" + DoubleToString(tpMultiplier, 1) + "_" + IntegerToString(p+9) + "_" + IntegerToString(countHiddenNeuron) + ".txt");
  }

void SaveThresoldsHidden(bool isLearning)
  {
   string data = "";
   for(int i=0;i<countHiddenNeuron;i++)
      data += thresoldsHidden[i] + "|=|";
   if(isLearning) WriteToFile(data, "L_Exp20_thresoldsHidden_" + IntegerToString(magicNumber) + "_" + IntegerToString(0) + "_" + DoubleToString(tpMultiplier, 1) + "_" + IntegerToString(p) + "_" + IntegerToString(countHiddenNeuron) + ".txt");
   else WriteToFile(data, "Exp20_thresoldsHidden_" + IntegerToString(magicNumber) + "_" + IntegerToString(0) + "_" + DoubleToString(tpMultiplier, 1) + "_" + IntegerToString(p) + "_" + IntegerToString(countHiddenNeuron) + ".txt");
  }

void SaveWeightsOutputLayer(bool isLearning)
  {
   string data = "";
   for(int c=0; c<countClasses; c++) {
      for(int i=0;i<countHiddenNeuron;i++) data += DoubleToString(weightsOutputLayer[c][i], 8) + "|=|";
      data += DoubleToString(thresoldOutputLayer[c], 8) + "|=|";
   }
   string name = (isLearning ? "L_Exp20_" : "Exp20_") + "weightsOut.txt";
   WriteToFile(data, name);
  }

void LoadWeightsOutputLayer(bool isLearning)
  {
   string fileName = (isLearning ? "L_Exp20_" : "Exp20_") + "weightsOut.txt";
   int handle = FileOpen(fileName, FILE_TXT|FILE_READ,';');
   if(handle<1) return;
   string data = FileReadString(handle);
   FileClose(handle);
   string elements[];
   StringSplit(data, '|', elements);
   int idx = 0;
   for(int c=0; c<countClasses; c++) {
      for(int i=0; i<countHiddenNeuron; i++) { weightsOutputLayer[c][i] = StringToDouble(elements[idx*2]); idx++; }
      thresoldOutputLayer[c] = StringToDouble(elements[idx*2]); idx++;
   }
  }

void WriteToFile(string data, string fileName)
  {
   FileDelete(fileName);
   int strLength = StringLen(data);
   int handle = FileOpen(fileName, FILE_TXT|FILE_READ|FILE_WRITE,';');
   if(handle<1)
     {
      Print(fileName, " Файл не обнаружен, последняя ошибка ", GetLastError());
     }
   else
     {
      FileWriteString(handle, data, strLength);
      FileClose(handle);
      Print("Данные успешно записаны в файл ", fileName);
     }  
  }

//+------------------------------------------------------------------+
//| Initialize pure arrays (Raw Prices & Deltas)                     |
//+------------------------------------------------------------------+
void InitBars()
  {
   for(int i=0; i<trainSize; i++)
     {
      int id = trainSize - i;
      double price = Close[id];
      double pricePrev = Close[id+1];
      
      rawPrices[i] = price;
      etalons[i] = price - pricePrev;
      
      double maF = iMA(NULL, 0, 14, 0, MODE_SMA, PRICE_CLOSE, id);
      double maF_prev = iMA(NULL, 0, 14, 0, MODE_SMA, PRICE_CLOSE, id+1);
      maFastNormArr[i] = ((maF - price) / _Point) / 1000.0;
      maFastSlopeArr[i] = ((maF - maF_prev) / _Point) / 10.0;
      
      double maS = iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE, id);
      double maS_prev = iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE, id+1);
      maSlowNormArr[i] = ((maS - price) / _Point) / 1000.0;
      maSlowSlopeArr[i] = ((maS - maS_prev) / _Point) / 10.0;
      
      atrArr[i] = (iATR(NULL, 0, 14, id) / _Point) / 1000.0;
      
      double angle = 2.0 * 3.14159265358979 * TimeHour(Time[id]) / 24.0;
      timeSinArr[i] = MathSin(angle);
      timeCosArr[i] = MathCos(angle);
     }
  }

//+------------------------------------------------------------------+
//| RSI Helper
//+------------------------------------------------------------------+
double CalcRSI(double &data[], int endIndex, int period)
  {
   if(endIndex < period) return 50.0;
   double sumGain = 0;
   double sumLoss = 0;
   for(int i = endIndex - period + 1; i <= endIndex; i++)
     {
      double diff = data[i];
      if(diff > 0) sumGain += diff;
      else sumLoss -= diff;
     }
   if(sumGain == 0 && sumLoss == 0) return 50.0;
   if(sumLoss == 0) return 100.0;
   double rs = sumGain / sumLoss;
   return 100.0 - (100.0 / (1.0 + rs));
  }

double CalculateLoss(int startIndex, int endIndex)
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
  }

//+------------------------------------------------------------------+
//| Train Network (Direct N-Step Prediction)                         |
//+------------------------------------------------------------------+
void Train()
  {
   
   if(allowSkipTrain == true && errGlobal < tradeErrThresold && fastPassCounter < skipTrainMaxPasses)
      return;
     
   fastPassCounter = 0;
   int maxSample = trainSize - p - predictHorizon - trainBuffer;

   for(int pox=0; pox < pass; pox++)
     {
      countTeaches++;
      errGlobal = 0;
      double grad_wOut[countClasses][countHiddenNeuron]; ArrayInitialize(grad_wOut, 0);
      double grad_tOut[countClasses]; ArrayInitialize(grad_tOut, 0);
      double grad_wHid[countHiddenNeuron][p+10]; ArrayInitialize(grad_wHid, 0);
      double grad_tHid[countHiddenNeuron]; ArrayInitialize(grad_tHid, 0);
      
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
         for(int c=0; c<countClasses; c++) {
            double sumOut = 0;
            for(int j=0; j<countHiddenNeuron; j++) sumOut += weightsOutputLayer[c][j] * outputsHidden[j];
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
         
         double targetVector[countClasses]; ArrayInitialize(targetVector, 0); targetVector[targetClass] = 1.0;
         targetVector[targetClass] = 1.0;
         
         double midExtreme = classExtremeThreshold * 1.5;
         double midStrong = (classExtremeThreshold + classStrongThreshold) / 2.0;
         double midWeak = (classStrongThreshold + classWeakThreshold) / 2.0;
         double expectedPoints = probsTrain[0] * (-midExtreme) + probsTrain[1] * (-midStrong) + probsTrain[2] * (-midWeak) + probsTrain[3] * 0 + probsTrain[4] * midWeak + probsTrain[5] * midStrong + probsTrain[6] * midExtreme;
         yValues[iSample] = expectedPoints * _Point;
         
         double weightPenalty = (targetClass == 3) ? penaltyFlat : 1.0;
         errGlobal += (-MathLog(probsTrain[targetClass] + 1e-10)) * weightPenalty;
         
         double errOutput[countClasses]; ArrayInitialize(errOutput, 0);
         for(int c=0; c<countClasses; c++) errOutput[c] = (probsTrain[c] - targetVector[c]) * weightPenalty;
         
         double errHiden[countHiddenNeuron] = {0};
         for(int j=0; j<countHiddenNeuron; j++)
           {
            double e = 0;
            for(int c=0; c<countClasses; c++) e += errOutput[c] * weightsOutputLayer[c][j];
            errHiden[j] = e;
           }
         for(int c=0; c<countClasses; c++)
           {
            for(int j=0; j<countHiddenNeuron; j++) grad_wOut[c][j] += errOutput[c] * outputsHidden[j];
            grad_tOut[c] += errOutput[c] * (-1.0);
           }

         for(int i=0; i<countHiddenNeuron; i++)
           {
            double common_grad = errHiden[i] * SigmoidDerivative(weightedSums[i]);
            for(int j=0; j<p; j++) grad_wHid[i][j] += common_grad * ((etalons[j + iSample] - localMin) / range);
            grad_wHid[i][p] += common_grad * ((range / _Point) / 1000.0);
            grad_wHid[i][p+1] += common_grad * ((CalcRSI(etalons, iSample + p - 1, 14) - 50.0) / 100.0);
            grad_wHid[i][p+2] += common_grad * ((CalcRSI(etalons, iSample + p - 1, 7) - 50.0) / 100.0);
            grad_wHid[i][p+3] += common_grad * timeSin;
            grad_wHid[i][p+4] += common_grad * maFastNorm;
            grad_wHid[i][p+5] += common_grad * maSlowNorm;
            grad_wHid[i][p+6] += common_grad * maFastSlope;
            grad_wHid[i][p+7] += common_grad * maSlowSlope;
            grad_wHid[i][p+8] += common_grad * atrArr[iSample + p - 1];
            grad_wHid[i][p+9] += common_grad * timeCos;
            grad_tHid[i] += common_grad * (-1.0);
           }
        }

      double numSamples = (double)(maxSample + 1);
      double a = aStep;
      adam_t++;
      double b1_t = 1.0 - MathPow(beta1, (double)adam_t);
      double b2_t = 1.0 - MathPow(beta2, (double)adam_t);
      
      for(int c=0; c<countClasses; c++)
        {
         for(int j=0; j<countHiddenNeuron; j++) {
            double g = grad_wOut[c][j] / numSamples;
            m_weightsOutputLayer[c][j] = beta1 * m_weightsOutputLayer[c][j] + (1 - beta1) * g;
            v_weightsOutputLayer[c][j] = beta2 * v_weightsOutputLayer[c][j] + (1 - beta2) * g * g;
            weightsOutputLayer[c][j] -= a * (m_weightsOutputLayer[c][j] / b1_t) / (MathSqrt(v_weightsOutputLayer[c][j] / b2_t) + epsilon);
         }
         double g_t = grad_tOut[c] / numSamples;
         m_thresoldOutputLayer[c] = beta1 * m_thresoldOutputLayer[c] + (1 - beta1) * g_t;
         v_thresoldOutputLayer[c] = beta2 * v_thresoldOutputLayer[c] + (1 - beta2) * g_t * g_t;
         thresoldOutputLayer[c] -= a * (m_thresoldOutputLayer[c] / b1_t) / (MathSqrt(v_thresoldOutputLayer[c] / b2_t) + epsilon);
        }

      for(int i=0; i<countHiddenNeuron; i++)
        {
         for(int j=0; j<=p+9; j++) {
            double g = grad_wHid[i][j] / numSamples;
            m_weightsHidden[i][j] = beta1 * m_weightsHidden[i][j] + (1 - beta1) * g;
            v_weightsHidden[i][j] = beta2 * v_weightsHidden[i][j] + (1 - beta2) * g * g;
            weightsHidden[i][j] -= a * (m_weightsHidden[i][j] / b1_t) / (MathSqrt(v_weightsHidden[i][j] / b2_t) + epsilon);
         }
         double g_t = grad_tHid[i] / numSamples;
         m_thresoldsHidden[i] = beta1 * m_thresoldsHidden[i] + (1 - beta1) * g_t;
         v_thresoldsHidden[i] = beta2 * v_thresoldsHidden[i] + (1 - beta2) * g_t * g_t;
         thresoldsHidden[i] -= a * (m_thresoldsHidden[i] / b1_t) / (MathSqrt(v_thresoldsHidden[i] / b2_t) + epsilon);
        }
        
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

   Print("Итерация " + countTeaches + " === Train Err: " + DoubleToString(errGlobal, 5) + " Val Err: " + DoubleToString(errValidation, 5));
   
   if(saveLearningProgress)
     {
      saveTimer++;
      if(saveTimer > saveTimerThresold)
        {
         saveTimer = 0;
         if(ovverideSaveFile) SaveWeightsData(false);
         else SaveWeightsData(true);
        }  
     }  
  }

//+------------------------------------------------------------------+
//| Predict Future                                                   |
//+------------------------------------------------------------------+
void Predict()
  {
   ArrayInitialize(predictedProbs, 0.0);

   int shiftBack = 0;
   int iSample = trainSize - 1 - shiftBack;
   
   if(iSample >= p)
     {
      double localMin = etalons[iSample - p + 1];
      double localMax = etalons[iSample - p + 1];
      for(int j=1; j<p; j++)
        {
         double val = etalons[iSample - p + 1 + j];
         if(val < localMin) localMin = val;
         if(val > localMax) localMax = val;
        }
      double range = localMax - localMin;
      if(range == 0) range = 0.00001;

      double window[p] = {0};
      for(int j=0; j<p; j++) window[j] = etalons[iSample - p + 1 + j];

      double outH[countHiddenNeuron] = {0};
      
      double currentPriceWin = rawPrices[iSample];
      double maFast = iMA(NULL, 0, 14, 0, MODE_SMA, PRICE_CLOSE, shiftBack + 1);
      double maFastPrev = iMA(NULL, 0, 14, 0, MODE_SMA, PRICE_CLOSE, shiftBack + 2);
      double maSlow = iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE, shiftBack + 1);
      double maSlowPrev = iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE, shiftBack + 2);
      
      double maFastNorm = ((maFast - currentPriceWin) / _Point) / 1000.0;
      double maSlowNorm = ((maSlow - currentPriceWin) / _Point) / 1000.0;
      double maFastSlope = ((maFast - maFastPrev) / _Point) / 10.0;
      double maSlowSlope = ((maSlow - maSlowPrev) / _Point) / 10.0;

      for(int i=0; i<countHiddenNeuron; i++)
        {
         double wSum = 0;
         for(int j=0; j<p; j++) wSum += weightsHidden[i][j] * ((window[j] - localMin) / range);
         wSum += weightsHidden[i][p] * ((range / _Point) / 1000.0);
         wSum += weightsHidden[i][p+1] * ((CalcRSI(etalons, iSample, 14) - 50.0) / 100.0);
         wSum += weightsHidden[i][p+2] * ((CalcRSI(etalons, iSample, 7) - 50.0) / 100.0);
         double predAngle = 2.0 * 3.14159265358979 * TimeHour(Time[shiftBack + 1]) / 24.0;
         wSum += weightsHidden[i][p+3] * MathSin(predAngle);
         wSum += weightsHidden[i][p+4] * maFastNorm;
         wSum += weightsHidden[i][p+5] * maSlowNorm;
         wSum += weightsHidden[i][p+6] * maFastSlope;
         wSum += weightsHidden[i][p+7] * maSlowSlope;
         wSum += weightsHidden[i][p+8] * ((iATR(NULL, 0, 14, shiftBack + 1) / _Point) / 1000.0);
         wSum += weightsHidden[i][p+9] * MathCos(predAngle);
         wSum -= thresoldsHidden[i];
         outH[i] = Sigmoid(wSum);
        }

      double outOutput[countClasses]; ArrayInitialize(outOutput, 0);
      for(int c=0; c<countClasses; c++)
        {
         double sumOut = 0;
         for(int i=0; i<countHiddenNeuron; i++) sumOut += weightsOutputLayer[c][i] * outH[i];
         sumOut -= thresoldOutputLayer[c];
         outOutput[c] = sumOut;
        }
        
      double maxOut = outOutput[0];
      for(int c=1; c<countClasses; c++) if(outOutput[c] > maxOut) maxOut = outOutput[c];
      double sumExp = 0;
      for(int c=0; c<countClasses; c++) { predictedProbs[c] = MathExp(outOutput[c] - maxOut); sumExp += predictedProbs[c]; }
      for(int c=0; c<countClasses; c++) { predictedProbs[c] /= sumExp; }
     }

   double midExtreme = classExtremeThreshold * 1.5;
   double midStrong = (classExtremeThreshold + classStrongThreshold) / 2.0;
   double midWeak = (classStrongThreshold + classWeakThreshold) / 2.0;

   double expectedPoints = predictedProbs[0] * (-midExtreme) + predictedProbs[1] * (-midStrong) + predictedProbs[2] * (-midWeak) + predictedProbs[3] * 0 + predictedProbs[4] * midWeak + predictedProbs[5] * midStrong + predictedProbs[6] * midExtreme;
   predictedDelta = expectedPoints * _Point;

   if(errGlobal < tradeErrThresold)
     {
      Trade();
     }
  }

void Trade()
  {
   RefreshRates();
   double pointsExpected = predictedDelta / _Point;
   int myOrders = 0;
   bool hasBuy = false;
   bool hasSell = false;
   
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
         if(OrderSymbol() == _Symbol && OrderMagicNumber() == magicNumber)
           {
            myOrders++;
            
            if(OrderType() == OP_BUY)
              {
               hasBuy = true;
               if(pointsExpected <= 0)
                 {
                  bool res = OrderClose(OrderTicket(), OrderLots(), Bid, 30, clrRed);
                  if(res) { hasBuy = false; myOrders--; }
                 }
              }
            else if(OrderType() == OP_SELL)
              {
               hasSell = true;
               if(pointsExpected >= 0)
                 {
                  bool res = OrderClose(OrderTicket(), OrderLots(), Ask, 30, clrRed);
                  if(res) { hasSell = false; myOrders--; }
                 }
              }
           }
        }
     }

   if(myOrders > 0 && onlyOneOrder == true)
     {
      ebobo = false;
      return;
     }

   ebobo = false;

   if(pointsExpected >= classExtremeThreshold && !hasBuy)
     {
      double sl = NormalizeDouble(Bid - (stopLoss * 1.5) * _Point, _Digits);
      double tp = NormalizeDouble(Bid + (takeProfit * 2.0) * _Point, _Digits);
      ticket = OrderSend(_Symbol, OP_BUY, 0.1, Ask, 30, sl, tp, "Extreme BUY", magicNumber);
     }
   else if(pointsExpected <= -classExtremeThreshold && !hasSell)
     {
      double sl = NormalizeDouble(Ask + (stopLoss * 1.5) * _Point, _Digits);
      double tp = NormalizeDouble(Ask - (takeProfit * 2.0) * _Point, _Digits);
      ticket = OrderSend(_Symbol, OP_SELL, 0.1, Bid, 30, sl, tp, "Extreme SELL", magicNumber);
     }
   else if(pointsExpected >= classStrongThreshold && !hasBuy)
     {
      double sl = NormalizeDouble(Bid - stopLoss * _Point, _Digits);
      double tp = NormalizeDouble(Bid + takeProfit * _Point, _Digits);
      ticket = OrderSend(_Symbol, OP_BUY, 0.1, Ask, 30, sl, tp, "Strong BUY", magicNumber);
     }
   else if(pointsExpected <= -classStrongThreshold && !hasSell)
     {
      double sl = NormalizeDouble(Ask + stopLoss * _Point, _Digits);
      double tp = NormalizeDouble(Ask - takeProfit * _Point, _Digits);
      ticket = OrderSend(_Symbol, OP_SELL, 0.1, Bid, 30, sl, tp, "Strong SELL", magicNumber);
     }
   else if(pointsExpected >= classWeakThreshold && !hasBuy)
     {
      double sl = NormalizeDouble(Bid - (stopLoss / 2.0) * _Point, _Digits);
      double tp = NormalizeDouble(Bid + (takeProfit / 2.0) * _Point, _Digits);
      ticket = OrderSend(_Symbol, OP_BUY, 0.1, Ask, 30, sl, tp, "Weak BUY", magicNumber);
     }
   else if(pointsExpected <= -classWeakThreshold && !hasSell)
     {
      double sl = NormalizeDouble(Ask + (stopLoss / 2.0) * _Point, _Digits);
      double tp = NormalizeDouble(Ask - (takeProfit / 2.0) * _Point, _Digits);
      ticket = OrderSend(_Symbol, OP_SELL, 0.1, Bid, 30, sl, tp, "Weak SELL", magicNumber);
     }
  }

//+------------------------------------------------------------------+
//| Manage Trailing Stop                                             |
//+------------------------------------------------------------------+
void ManageTrailingStop()
  {
   if(!useTrailingStop) return;
   
   double minStop = MarketInfo(_Symbol, MODE_STOPLEVEL);
   double tStart = (trailingStart < minStop) ? minStop : trailingStart;
   
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
         if(OrderSymbol() == _Symbol && OrderMagicNumber() == magicNumber)
           {
            if(OrderType() == OP_BUY)
              {
               if(Bid - OrderOpenPrice() > tStart * _Point)
                 {
                  double newSL = NormalizeDouble(Bid - tStart * _Point, _Digits);
                  if(OrderStopLoss() < newSL - trailingStep * _Point || OrderStopLoss() == 0)
                    {
                     bool res = OrderModify(OrderTicket(), OrderOpenPrice(), newSL, OrderTakeProfit(), 0, clrGreen);
                    }
                 }
              }
            else if(OrderType() == OP_SELL)
              {
               if(OrderOpenPrice() - Ask > tStart * _Point)
                 {
                  double newSL = NormalizeDouble(Ask + tStart * _Point, _Digits);
                  if(OrderStopLoss() > newSL + trailingStep * _Point || OrderStopLoss() == 0)
                    {
                     bool res = OrderModify(OrderTicket(), OrderOpenPrice(), newSL, OrderTakeProfit(), 0, clrRed);
                    }
                 }
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Main Tick Event                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(Bars < trainSize + 50) return;
   
   ManageTrailingStop();
   
   if(IsNewBar())
     {
      // --- Weekly Reset Logic ---
      static int lastResetDay = -1;
      if(TimeDayOfWeek(Time[0]) == 1 && TimeDay(Time[0]) != lastResetDay)
        {
         lastResetDay = TimeDay(Time[0]);
         if(useFixedSeed) MathSrand(randomSeed);
         InitWeights();
         Print("Weekly Reset: Weights and Adam optimizer reset to fixed seed!");
         
         for(int i = OrdersTotal() - 1; i >= 0; i--)
           {
            if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
              {
               if(OrderSymbol() == _Symbol && OrderMagicNumber() == magicNumber)
                 {
                  if(OrderType() == OP_BUY) OrderClose(OrderTicket(), OrderLots(), Bid, 30, clrRed);
                  if(OrderType() == OP_SELL) OrderClose(OrderTicket(), OrderLots(), Ask, 30, clrRed);
                 }
              }
           }
        }
      // --------------------------

      fastPassCounter++;
      nonTeacheble = false;
      ebobo = true;
      bestErrValidation = 999999;
      earlyStopCounter = 0;
      InitBars();
      if(allowSkipTrain) CalculateError();
     }

   if(ebobo && !nonTeacheble)
     {
      Train();
      Predict();
      CleanupGraphics();
      DrawLines();
      DrawLinesNN();
      DrawLinesPredict();
     }

   string status = (errGlobal < tradeErrThresold) ? "[OK] READY (Trading)" : "[!] LEARNING";
   color txtColor = infoTextColor;
   if(txtColor == clrNONE) txtColor = (color)ChartGetInteger(0, CHART_COLOR_FOREGROUND);
   ChartSetInteger(0, CHART_FOREGROUND, false);
   
   string lines[15];
   lines[0] = "--- Antigravity NN v21 ---";
   lines[1] = "Status: " + status;
   lines[2] = "Arch: Inp(" + IntegerToString(p+10) + ") -> Hid(" + IntegerToString(countHiddenNeuron) + ") -> Out(Softmax)";
   lines[3] = "Mem: " + IntegerToString(trainSize) + " bars | Train buf: " + IntegerToString(trainBuffer) + " | +ATR +sin/cos(time)";
   lines[4] = "Train Err: " + DoubleToString(errGlobal, 5) + " | Val Err: " + DoubleToString(errValidation, 5);
   lines[5] = "Passes Done: " + IntegerToString(countTeaches);
   lines[6] = "Pass (Epochs per tick): " + IntegerToString(pass);
   lines[7] = "Learning Rate (aStep): " + DoubleToString(aStep, 6);
   lines[8] = "--- Trading Logic ---";
   lines[9] = "Predict Horizon: " + IntegerToString(predictHorizon) + " bars ahead";
   lines[10] = "Probs [E.Sell, S.Sell, W.Sell, Flat, W.Buy, S.Buy, E.Buy]";
   lines[11] = StringFormat("[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]", predictedProbs[0], predictedProbs[1], predictedProbs[2], predictedProbs[3], predictedProbs[4], predictedProbs[5], predictedProbs[6]);
   lines[12] = "Prob Trade Threshold: " + DoubleToString(probTradeThreshold, 2);
   lines[13] = "Stop Loss: " + IntegerToString(stopLoss) + " points";
   lines[14] = "Take Profit Limit: " + IntegerToString(takeProfit) + " points";
   
   for(int k = 0; k < 15; k++)
     {
      string objName = "InfoLabel_" + IntegerToString(k);
      if(ObjectFind(objName) < 0)
        {
         ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0);
         ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
        }
      ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, 20 + k * lineSpacing);
      ObjectSetString(0, objName, OBJPROP_TEXT, lines[k]);
      ObjectSetString(0, objName, OBJPROP_FONT, "Arial");
      ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, fontSize);
      ObjectSetInteger(0, objName, OBJPROP_COLOR, txtColor);
      ObjectSetInteger(0, objName, OBJPROP_BACK, false);
     }
   Comment("");
  }

//+------------------------------------------------------------------+
//| Drawing Lines                                                    |
//+------------------------------------------------------------------+
void DrawLines()
  {
   ObjectsDeleteAll(0, "Ebat_");
   for(int i=0; i<trainSize-1; i++)
     {
      int barIndex1 = trainSize - i - 1;
      int barIndex2 = MathMax(trainSize - i - 2, 0);
      
      double x1 = Time[barIndex1];
      double y1 = rawPrices[i] + offsetY;
      double x2 = Time[barIndex2];
      double y2 = rawPrices[i+1] + offsetY;
      
      string objName = "Ebat_" + IntegerToString(i);
      ObjectCreate(0, objName, OBJ_TREND, 0, x1, y1, x2, y2);
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clrBlue);
      ObjectSetInteger(0, objName, OBJPROP_WIDTH, 5);
      ObjectSetInteger(0, objName, OBJPROP_RAY, 0);
      ObjectSetInteger(0, objName, OBJPROP_BACK, false);
     }
  }

void DrawLinesNN()
  {
   ObjectsDeleteAll(0, "NN_");
   int maxSample = trainSize - p - predictHorizon;
   
   for(int i=0; i < maxSample - 1; i++)
     {
      int bar1 = trainSize - (i + p - 1 + predictHorizon) - 1;
      int bar2 = trainSize - (i + 1 + p - 1 + predictHorizon) - 1;

      if(bar1 < 0 || bar2 < 0) continue;

      double x1 = Time[bar1];
      double y1 = rawPrices[i + p - 1] + yValues[i] + offsetY; 

      double x2 = Time[bar2];
      double y2 = rawPrices[i + 1 + p - 1] + yValues[i+1] + offsetY;

      string objName = "NN_" + IntegerToString(i);
      ObjectCreate(0, objName, OBJ_TREND, 0, x1, y1, x2, y2);
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clrYellow);
      ObjectSetInteger(0, objName, OBJPROP_WIDTH, 3);
      ObjectSetInteger(0, objName, OBJPROP_RAY, 0);
      ObjectSetInteger(0, objName, OBJPROP_BACK, false);
     }
  }

void DrawLinesPredict()
  {
   ObjectsDeleteAll(0, "PredictLine");
   double x1 = Time[0];
   double y1 = rawPrices[trainSize - 1] + offsetY; 
   
   datetime futureTime = Time[0] + PeriodSeconds() * predictHorizon;
   double y2 = y1 + predictedDelta;

   ObjectCreate(0, "PredictLine", OBJ_TREND, 0, x1, y1, futureTime, y2);
   ObjectSetInteger(0, "PredictLine", OBJPROP_COLOR, clrLime);
   ObjectSetInteger(0, "PredictLine", OBJPROP_WIDTH, 3);
   ObjectSetInteger(0, "PredictLine", OBJPROP_RAY, 0);
   ObjectSetInteger(0, "PredictLine", OBJPROP_BACK, false);
  }

//+------------------------------------------------------------------+
//| Math helpers                                                     |
//+------------------------------------------------------------------+
double RandomDouble(double min, double max)
  {
   return min + (max - min) * MathRand() / 32767.0;
  }

double Sigmoid(double x)
  {
   if(x > 0) return x;
   else return 0.01 * x;
  }

double SigmoidDerivative(double y)
  {
   if(y > 0) return 1;
   else return 0.01;
  }

bool IsNewBar()
  {
   static int nBars = 0;
   if(nBars != Bars)
     {
      nBars = Bars;
      return(true);
     }
   return(false);
  }
//+------------------------------------------------------------------+
