//+------------------------------------------------------------------+
//|                                                         NN_5.mq4 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#define p 48//64
#define countHiddenNeuron 6//24
#define trainSize 1536
#define predictBars 24

input double tpMultiplier = 1.0;
input bool readInitData = false;
input int takeProfit = 300;
input int stopLoss = 400;
input int orderThresold = 150;
input int pass = 30;
int maxTeach = 1000;
input int countPredictAnalysys = 6;
input int predictShift = 3;
input double wRange = 0.35;// диапозон случайных значений весов 0.9;
input bool isRealtime;
input bool saveLearningProgress;
input bool ovverideSaveFile;
input int saveTimerThresold = 100;
input bool allowSkipTrain = false;
input int skipTrainMaxPasses = 24;
input double aStep = 0.01;
input double momentum = 0.9; // Инерция для SGD
input bool onlyOneOrder = true;
input bool closeOnReverseSignal = true;
input int magicNumber = 123456;
input bool useRandomSeed = false;  // использовать фиксированный seed для генератора
input int randomSeed = 42;         // значение seed (активно только если useRandomSeed = true)

input int checkLastNTrades = 5;          // количество последних сделок для анализа
input int maxLossThresholdPips = 1000;   // порог суммарного убытка (в пунктах) для сброса весов

datetime lastResetTime = 0;

double weightsHidden[countHiddenNeuron][p+2];
double thresoldsHidden[countHiddenNeuron];
double weightedSums[countHiddenNeuron];
double outputsHidden[countHiddenNeuron];
double weightsOutputLayer[countHiddenNeuron];
double thresoldOutputLayer;

double vWeightsHidden[countHiddenNeuron][p+2];
double vThresoldsHidden[countHiddenNeuron];
double vWeightsOutputLayer[countHiddenNeuron];
double vThresoldOutputLayer;

double yValues[trainSize - p];
double etalons[trainSize];
double rawPrices[trainSize];


double offsetY = 0;// 0.195;
int countTeaches = 0;
double errIncrease = 0;

int saveTimer = 0;
int ticket;

bool ebobo;
bool nonTeacheble;
double errGlobal = 999;
int countTryed;

input double tradeErrThresold = 0.235;// Порог ошибки для торговли 0.13
double teachErrThresold = 3.5;
int fastPassCounter;

double predictValues[predictBars];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   ObjectsDeleteAll();
//---
   InitBars();
   InitWeights();
//---
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void InitWeights()
  {
   countTeaches = 0;
   errGlobal = 999;

   if(readInitData)
     {
      string strWeightsHidden;
      int handle;
      handle = FileOpen("Exp7_weightsHidden.txt", FILE_TXT|FILE_READ);
      if(handle > 0)
        {
         strWeightsHidden = FileReadString(handle);
         Print(StringReplace(strWeightsHidden, "|=|", "=") + " Было произведено замен");
         string sep = "=";
         ushort u_sep;
         string result[];
         u_sep = StringGetCharacter(sep,0);
         int k = StringSplit(strWeightsHidden, u_sep, result) - 1;
         if(k != countHiddenNeuron * (p+2))
           {
            Print("!!!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!!");
           }
         else
           {
            int idxRead = 0;
             for(int i=0; i < countHiddenNeuron; i++)
               {
                for(int j=0; j <= p+1; j++)
                    {
                     weightsHidden[i][j] = StrToDouble(result[idxRead]);
                    idxRead++;
                 }
              }
           }
           FileClose(handle);
        }
      else
        {
         Print("Ebala Syet ************ " + GetLastError());
        }
      
      //==============================================================================
      string strThresoldsHidden;
      handle = FileOpen("Exp7_thresoldsHidden.txt", FILE_TXT|FILE_READ);
      if(handle > 0)
      {
         strThresoldsHidden = FileReadString(handle);
         Print(StringReplace(strThresoldsHidden, "|=|", "=") + " Было произведено замен");
         string result[];
         int k = StringSplit(strThresoldsHidden, StringGetCharacter("=", 0), result) - 1;
         if(k != countHiddenNeuron)
         {
            Print("!!!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!!");
         }
         else
         {
            int idxRead = 0;
            for(int i=0; i<countHiddenNeuron; i++)
            {
               thresoldsHidden[i] = StrToDouble(result[idxRead]);
               idxRead++;
            }
         }
         FileClose(handle);
      }
      
      //==============================================================================
      string strWeightsOutputLayer;
      handle = FileOpen("Exp7_weightsOutputLayer.txt", FILE_TXT|FILE_READ);
      if(handle > 0)
      {
         strWeightsOutputLayer = FileReadString(handle);
         Print(StringReplace(strWeightsOutputLayer, "|=|", "=") + " Было произведено замен");
         string result[];
         int k = StringSplit(strWeightsOutputLayer, StringGetCharacter("=", 0), result) - 1;
         if(k == countHiddenNeuron)
         {
            int idxRead = 0;
            for(int i=0; i<countHiddenNeuron; i++)
            {
               weightsOutputLayer[i] = StrToDouble(result[idxRead]);
               idxRead++;
            }
            thresoldOutputLayer = 0.0;
         }
         else if(k == countHiddenNeuron + 1)
         {
            int idxRead = 0;
            for(int i=0; i<countHiddenNeuron; i++)
            {
               weightsOutputLayer[i] = StrToDouble(result[idxRead]);
               idxRead++;
            }
            thresoldOutputLayer = StrToDouble(result[idxRead]);
         }
         else
         {
            Print("!!!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!!");
         }
         FileClose(handle);
      }
        Print("Successfully loaded weights from files!");
     }
     else
     {
      if(useRandomSeed) MathSrand(randomSeed);
      else              MathSrand(GetTickCount());
      string opta = "";
      for(int i=0; i<countHiddenNeuron; i++)
      {
         for(int j=0; j<=p+1; j++)
           {
              weightsHidden[i][j] = RandomDouble(-wRange, wRange);
              opta += weightsHidden[i][j] + "|=|";
           }
         thresoldsHidden[i] = RandomDouble(-wRange, wRange);
         weightsOutputLayer[i] = RandomDouble(-wRange, wRange);
      }
      thresoldOutputLayer = RandomDouble(-wRange, wRange);
      WriteToFile(opta, "Exp7_weightsHidden.txt");
   
      SaveThresoldsHidden(false);
      SaveWeightsOutputLayer(false);
      Print("======================================= Веса Инициализированы ===================================");
     }
     
   for(int i=0; i<countHiddenNeuron; i++)
     {
      for(int j=0; j<=p+1; j++)
        {
         vWeightsHidden[i][j] = 0.0;
        }
      vThresoldsHidden[i] = 0.0;
      vWeightsOutputLayer[i] = 0.0;
     }
   vThresoldOutputLayer = 0.0;
  }
  
//----------------------------------------------------------------------  
void SaveWeightsData(bool isLearning)
{
   SaveWeightsHiddenLayer(isLearning);
   SaveThresoldsHidden(isLearning);
   SaveWeightsOutputLayer(isLearning);
}

//----------------------------------------------------------------------
void SaveWeightsHiddenLayer(bool isLearning)
{
   string data = "";
   for(int i=0; i<countHiddenNeuron; i++)
   {
      for(int j=0; j<=p+1; j++)
      {
         data += weightsHidden[i][j] + "|=|";
      }
   }
   if(isLearning) WriteToFile(data, "L_Exp7_weightsHidden.txt");
   else WriteToFile(data, "Exp7_weightsHidden.txt");
}

//----------------------------------------------------------------------
void SaveThresoldsHidden(bool isLearning)
{
   string data = "";
   for(int i=0;i<countHiddenNeuron;i++)
     {
      data += thresoldsHidden[i] + "|=|";
     }
   if(isLearning) WriteToFile(data, "L_Exp7_thresoldsHidden.txt");
   else WriteToFile(data, "Exp7_thresoldsHidden.txt");
}

//-----------------------------------------------------------------------
void SaveWeightsOutputLayer(bool isLearning)
{
   string data = "";
   for(int i=0;i<countHiddenNeuron;i++)
     {
      data += weightsOutputLayer[i] + "|=|";
     }
   data += thresoldOutputLayer + "|=|";
   if(isLearning) WriteToFile(data, "L_Exp7_weightsOutputLayer.txt");
   else WriteToFile(data, "Exp7_weightsOutputLayer.txt");
}

//-----------------------------------------------------------------------
void WriteToFile(string data, string fileName)
{
   FileDelete(fileName);
   int strLength = StringLen(data);
   int handle;
   handle = FileOpen(fileName, FILE_TXT|FILE_READ|FILE_WRITE,';');
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
//| Calculate pure arrays (Without Normalization look-ahead)         |
//+------------------------------------------------------------------+
void InitBars()
  {
   for(int i=0; i<trainSize; i++)
     {
      int id = trainSize - i;
      double price = Close[id];
      double pricePrev = Close[id+1];
      
      rawPrices[i] = price;
      
      double ret = price - pricePrev;
      etalons[i] = ret;
      
     }
  }

//+------------------------------------------------------------------+
//| Calculate Error without modifying weights                        |
//+------------------------------------------------------------------+
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

void CalculateError()
  {
   errGlobal = 0;
   for(int iSample=0; iSample<trainSize-p; iSample++)
     {
      // --- Rolling Window Normalization ---
      double localMin = 999999;
      double localMax = -999999;
      for(int j=0; j<p; j++)
        {
         if(etalons[j+iSample] < localMin) localMin = etalons[j+iSample];
         if(etalons[j+iSample] > localMax) localMax = etalons[j+iSample];
        }
      double range = localMax - localMin;
      if(range < 100 * _Point) range = 100 * _Point;
      // ------------------------------------

      double rsi = CalcRSI(etalons, iSample + p - 1, 14);
      double rsiInput = (rsi - 50.0) / 100.0;
      double rsi2 = CalcRSI(etalons, iSample + p - 1, 28);
      double rsiInput2 = (rsi2 - 50.0) / 100.0;

      for(int i=0; i<countHiddenNeuron; i++)
        {
         double wSum = 0;
         for(int j=0; j<p; j++)
           {
            double x = (etalons[j+iSample] - localMin) / range;
            wSum += weightsHidden[i][j] * x;
           }
         
         wSum += weightsHidden[i][p] * rsiInput;
         wSum += weightsHidden[i][p+1] * rsiInput2;
         
         wSum -= thresoldsHidden[i];
         weightedSums[i] = wSum;
         outputsHidden[i] = LeakyReLU(wSum);
        }

      double weightedSumOutputLayer = 0;
      for(int i=0; i<countHiddenNeuron; i++)
        {
         double x = outputsHidden[i];
         weightedSumOutputLayer += weightsOutputLayer[i] * x;
        }
      weightedSumOutputLayer -= thresoldOutputLayer;
      double outputMain = weightedSumOutputLayer; // Linear Activation

      // Исправление Бага 3: обновляем массив предсказаний при пропуске обучения
      yValues[iSample] = outputMain * range + localMin;

      // Целевое значение тоже нормализуется относительно текущего окна
      double targetNorm = (etalons[iSample + p] - localMin) / range;
      double errOutput = outputMain - targetNorm;
      
      // Возвращаем ошибку к реальному масштабу цен окна
      double errPrice = errOutput * range; 
      errGlobal += errPrice * errPrice;
     }
   errGlobal = (errGlobal / (trainSize - p)) * 100000;
  }

//+------------------------------------------------------------------+
//| Train with Rolling Window                                        |
//+------------------------------------------------------------------+
void Train()
  {
   
   if(allowSkipTrain == true && errGlobal < tradeErrThresold && fastPassCounter < skipTrainMaxPasses)
     {
      return;
     }
     
   fastPassCounter = 0;

   for(int pox=0; pox < pass; pox++)
     {
      countTeaches++;
      errGlobal = 0;

      for(int iSample=0; iSample<trainSize-p; iSample++)
        {
         // --- Rolling Window Normalization ---
         double localMin = 999999;
         double localMax = -999999;
         for(int j=0; j<p; j++)
           {
            if(etalons[j+iSample] < localMin) localMin = etalons[j+iSample];
            if(etalons[j+iSample] > localMax) localMax = etalons[j+iSample];
           }
         double range = localMax - localMin;
         if(range < 100 * _Point) range = 100 * _Point;
         // ------------------------------------

         // Исправление Бага 1: выносим вычисление RSI из цикла по нейронам
         double rsi = CalcRSI(etalons, iSample + p - 1, 14);
         double rsiInput = (rsi - 50.0) / 100.0;

         double rsi2 = CalcRSI(etalons, iSample + p - 1, 28);
         double rsiInput2 = (rsi2 - 50.0) / 100.0;

         for(int i=0; i<countHiddenNeuron; i++)
           {
            double wSum = 0;
            for(int j=0; j<p; j++)
            {
               double x = (etalons[j+iSample] - localMin) / range;
               wSum += weightsHidden[i][j] * x;
            }
            
            wSum += weightsHidden[i][p] * rsiInput;
            wSum += weightsHidden[i][p+1] * rsiInput2;
            
            wSum -= thresoldsHidden[i];
            weightedSums[i] = wSum;
            outputsHidden[i] = LeakyReLU(wSum);
           }

         double weightedSumOutputLayer = 0;
         for(int i=0; i<countHiddenNeuron; i++)
           {
            double x = outputsHidden[i];
            weightedSumOutputLayer += weightsOutputLayer[i] * x;
           }
         weightedSumOutputLayer -= thresoldOutputLayer;
         double outputMain = weightedSumOutputLayer; // Linear Activation

         // Денормализация для графики
         yValues[iSample] = outputMain * range + localMin;

         double a = aStep;
         double targetNorm = (etalons[iSample + p] - localMin) / range;
         double errOutput = outputMain - targetNorm;
         double errPrice = errOutput * range;
         errGlobal += errPrice * errPrice;
         
         // Умножаем на range для корректного градиента цены, 
         // и делим на 100 * _Point для сохранения масштаба скорости обучения (aStep)
         double gradScale = range / (100 * _Point);
         
         double errHiden[countHiddenNeuron];
         for(int i=0; i<countHiddenNeuron; i++)
           {
            errHiden[i] = errOutput * gradScale * weightsOutputLayer[i];
           }

         for(int i=0; i<countHiddenNeuron; i++)
           {
            double grad = errOutput * gradScale * outputsHidden[i];
            vWeightsOutputLayer[i] = momentum * vWeightsOutputLayer[i] - a * grad;
            weightsOutputLayer[i] += vWeightsOutputLayer[i];
           }

         double gradThresoldOut = -errOutput * gradScale;
         vThresoldOutputLayer = momentum * vThresoldOutputLayer - a * gradThresoldOut;
         thresoldOutputLayer += vThresoldOutputLayer;

         for(int i=0; i<countHiddenNeuron; i++)
           {
            for(int j=0; j<p; j++)
              {
               double x = (etalons[j + iSample] - localMin) / range;
               double grad = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * x;
               vWeightsHidden[i][j] = momentum * vWeightsHidden[i][j] - a * grad;
               weightsHidden[i][j] += vWeightsHidden[i][j];
              }
              
              double gradRsi = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * rsiInput;
              vWeightsHidden[i][p] = momentum * vWeightsHidden[i][p] - a * gradRsi;
              weightsHidden[i][p] += vWeightsHidden[i][p];

              double gradRsi2 = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * rsiInput2;
              vWeightsHidden[i][p+1] = momentum * vWeightsHidden[i][p+1] - a * gradRsi2;
              weightsHidden[i][p+1] += vWeightsHidden[i][p+1];
              
              double gradThresoldHid = -errHiden[i] * LeakyReLUDerivative(weightedSums[i]);
              vThresoldsHidden[i] = momentum * vThresoldsHidden[i] - a * gradThresoldHid;
              thresoldsHidden[i] += vThresoldsHidden[i];
           }
        }
      errGlobal = (errGlobal / (trainSize - p)) * 100000;
     }

   Print("Итерация " + countTeaches + " === Ошибко " + errGlobal + " Save Timer " + saveTimer);
   if(errGlobal > teachErrThresold && readInitData == false)
     {
      InitWeights();
      countTryed++;
     }
     
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
//| Predict with Dynamic Window Bounds                               |
//+------------------------------------------------------------------+
void Predict()
  {
   for(int pre=0; pre<predictBars; pre++)
     {
      // Формируем окно для текущего предсказания и находим его мин/макс
      double localMin = 999999;
      double localMax = -999999;
      double window[p];
      
      for(int j=0; j<p; j++)
        {
         int id = trainSize - p + j + pre;
         if(id >= trainSize)
           {
            int predictIndex = id - trainSize;
            window[j] = predictValues[predictIndex];
           }
         else
           {
            window[j] = etalons[id];
           }
           
         if(window[j] < localMin) localMin = window[j];
         if(window[j] > localMax) localMax = window[j];
        }
        
      double range = localMax - localMin;
      if(range < 100 * _Point) range = 100 * _Point;

      double outH[countHiddenNeuron];
      
      double rsi = CalcRSI(window, p - 1, 14);
      double rsiInput = (rsi - 50.0) / 100.0;
      double rsi2 = CalcRSI(window, p - 1, 28);
      double rsiInput2 = (rsi2 - 50.0) / 100.0;

      for(int i=0; i<countHiddenNeuron; i++)
        {
         double wSum = 0;
         for(int j=0; j<p; j++)
           {
            double x = (window[j] - localMin) / range;
            wSum += weightsHidden[i][j] * x;
           }
         
         wSum += weightsHidden[i][p] * rsiInput;
         wSum += weightsHidden[i][p+1] * rsiInput2;
         
         wSum -= thresoldsHidden[i];
         outH[i] = LeakyReLU(wSum);
        }

      double weightedSumOutputLayer = 0;
      for(int i=0; i<countHiddenNeuron; i++)
        {
         double x = outH[i];
         weightedSumOutputLayer += weightsOutputLayer[i] * x;
        }
      weightedSumOutputLayer -= thresoldOutputLayer;
      double outputMain = weightedSumOutputLayer; // Linear Activation
      
      // Денормализуем предсказание, возвращая его в масштаб цены окна
      predictValues[pre] = outputMain * range + localMin;
     }

   if(errGlobal < tradeErrThresold)
     {
      if(isRealtime)
      {
         if(ebobo) Trade();
      }
      else
      {
         Trade();
      }
     }
  }


//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void Trade()
  {
   double curPrice = iClose(NULL, 0, 0);

   double sumLow = 0;
   double sumHigh = 0;
   int countLow = 0;
   int countHigh = 0;

   int total = predictBars > countPredictAnalysys ? countPredictAnalysys : predictBars;
   double predictedPrice = rawPrices[trainSize - 1];

   for(int j=0; j<predictShift; j++)
     {
      if(j < predictBars) predictedPrice += predictValues[j];
     }

   for(int i=0; i<total; i++)
     {
      int idx = i + predictShift;
      if(idx >= predictBars) break;
      predictedPrice += predictValues[idx];
      if(predictedPrice > curPrice)
        {
         countHigh++;
         sumHigh += predictedPrice;
        }
      else
        {
         countLow++;
         sumLow += predictedPrice;
        }
     }

   int pipsHigh = 0;
   int pipsLow = 0;
   if(countHigh > 0)
     {
      double averageHigh = sumHigh / (double)countHigh;
      pipsHigh = (averageHigh - curPrice) / _Point;
     }
   if(countLow > 0)
     {
      double averageLow = sumLow / (double)countLow;
      pipsLow = (curPrice - averageLow) / _Point;
     }

   bool isBuySignal = (pipsHigh - pipsLow > orderThresold);
   bool isSellSignal = (pipsLow - pipsHigh > orderThresold);

   if(closeOnReverseSignal && OrdersTotal() > 0)
     {
      for(int i=OrdersTotal()-1; i>=0; i--)
        {
         if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
           {
            if(OrderSymbol() == _Symbol && OrderMagicNumber() == magicNumber)
              {
               if(OrderType() == OP_BUY && isSellSignal)
                 {
                  bool res = OrderClose(OrderTicket(), OrderLots(), Bid, 5);
                 }
               else if(OrderType() == OP_SELL && isBuySignal)
                 {
                  bool res = OrderClose(OrderTicket(), OrderLots(), Ask, 5);
                 }
              }
           }
        }
     }

   if(OrdersTotal() > 0 && onlyOneOrder == true)
     {
      ebobo = false;
      return;
     }

   ebobo = false;
   if(isBuySignal)
     {
      int t = pipsHigh > takeProfit ? takeProfit : pipsHigh * tpMultiplier;
      int s = pipsLow < stopLoss ? stopLoss : pipsLow;
      double sl = Bid - s * _Point;
      double tp = Bid + t *_Point;
      ticket = OrderSend(_Symbol, OP_BUY, 0.1, Ask, 5, sl, tp, "", magicNumber);
     }
   else if(isSellSignal)
     {
      int t = pipsLow > takeProfit ? takeProfit : pipsLow * tpMultiplier;
      int s = pipsHigh < stopLoss ? stopLoss : pipsHigh;
      double sl = Ask + s * _Point;
      double tp = Ask - t *_Point;
      ticket = OrderSend(_Symbol, OP_SELL, 0.1, Bid, 5, sl, tp, "", magicNumber);
     }
  }

//+------------------------------------------------------------------+
//| Check Drawdown and Reset Weights                                 |
//+------------------------------------------------------------------+
void CheckDrawdownAndReset()
  {
   if(checkLastNTrades <= 0) return;
   
   int historyTotal = OrdersHistoryTotal();
   if(historyTotal == 0) return;
   
   int n = checkLastNTrades;
   datetime latestTimes[];
   double latestProfits[];
   
   ArrayResize(latestTimes, n);
   ArrayResize(latestProfits, n);
   ArrayInitialize(latestTimes, 0);
   
   for(int i = 0; i < historyTotal; i++)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
        {
         if(OrderSymbol() == _Symbol && OrderMagicNumber() == magicNumber)
           {
            int type = OrderType();
            if(type == OP_BUY || type == OP_SELL)
              {
               datetime t = OrderCloseTime();
               
               // Игнорируем сделки, которые закрылись до или во время последнего сброса
               if(t <= lastResetTime) continue;
               
               if(t > latestTimes[n-1])
                 {
                  int insertPos = n - 1;
                  while(insertPos > 0 && t > latestTimes[insertPos-1])
                    {
                     insertPos--;
                    }
                    
                  for(int j = n - 1; j > insertPos; j--)
                    {
                     latestTimes[j] = latestTimes[j-1];
                     latestProfits[j] = latestProfits[j-1];
                    }
                    
                  latestTimes[insertPos] = t;
                  
                  if(type == OP_BUY) latestProfits[insertPos] = (OrderClosePrice() - OrderOpenPrice()) / _Point;
                  else               latestProfits[insertPos] = (OrderOpenPrice() - OrderClosePrice()) / _Point;
                 }
              }
           }
        }
     }
     
   double totalProfitPips = 0;
   int tradesFound = 0;
   
   for(int i = 0; i < n; i++)
     {
      if(latestTimes[i] > 0)
        {
         totalProfitPips += latestProfits[i];
         tradesFound++;
        }
     }
     
   if(tradesFound > 0 && totalProfitPips <= -maxLossThresholdPips)
     {
      Print("Суммарный убыток последних " + IntegerToString(tradesFound) + " сделок составил " + DoubleToString(-totalProfitPips, 1) + " пунктов. Сброс весов!");
      lastResetTime = latestTimes[0]; // Запоминаем время закрытия самой последней сделки в серии
      InitWeights();
      
      // Принудительное закрытие всех сделок (Panic Button)
      for(int k = OrdersTotal() - 1; k >= 0; k--)
        {
         if(OrderSelect(k, SELECT_BY_POS, MODE_TRADES))
           {
            if(OrderSymbol() == _Symbol && OrderMagicNumber() == magicNumber)
              {
               if(OrderType() == OP_BUY)
                 {
                  bool res = OrderClose(OrderTicket(), OrderLots(), Bid, 5);
                 }
               else if(OrderType() == OP_SELL)
                 {
                  bool res = OrderClose(OrderTicket(), OrderLots(), Ask, 5);
                 }
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   CheckDrawdownAndReset();

   if(IsNewBar())
     {
      countTryed = 0;
      nonTeacheble = false;
      ebobo = true;
      InitBars();
      if(allowSkipTrain) CalculateError();
      fastPassCounter++;
     }

   if(ebobo && !nonTeacheble)
     {
      Train();
      Predict();
      DrawLines();
      DrawLinesNN();
      DrawLinesPredict();
     }
     
  if(isRealtime)
    {
      Train();
      Predict();
      DrawLines();
      DrawLinesNN();
      DrawLinesPredict();
    }

   string status = (errGlobal < tradeErrThresold) ? "[OK] READY (Trading)" : "[!] LEARNING";
   color txtColor = (color)ChartGetInteger(0, CHART_COLOR_FOREGROUND);
   ChartSetInteger(0, CHART_FOREGROUND, false);
   string lines[13];
   lines[0] = "--- Antigravity NN v7 ---";
   lines[1] = "Status: " + status;
   lines[2] = "Arch: Inp(" + IntegerToString(p+1) + ") -> Hid(" + IntegerToString(countHiddenNeuron) + ") -> Out(Linear)";
   lines[3] = "Mem (trainSize): " + IntegerToString(trainSize) + " bars | RSI: Enabled (Dynamic)";
   lines[4] = "Global Error: " + DoubleToString(errGlobal, 5) + " / " + DoubleToString(tradeErrThresold, 5);
   lines[5] = "Passes Done: " + IntegerToString(countTeaches);
   lines[6] = "Pass (Epochs per tick): " + IntegerToString(pass);
   lines[7] = "Learning Rate (aStep): " + DoubleToString(aStep, 6);
   lines[8] = "--- Trading Logic ---";
   lines[9] = "Trade Horizon: " + IntegerToString(countPredictAnalysys) + " (Vis: " + IntegerToString(predictBars) + ") | Shift: " + IntegerToString(predictShift);
   lines[10] = "Order Threshold: " + IntegerToString(orderThresold) + " points";
   lines[11] = "Stop Loss: " + IntegerToString(stopLoss) + " points";
   lines[12] = "Take Profit Limit: " + IntegerToString(takeProfit) + " points";
   
   for(int k = 0; k < 13; k++)
     {
      string objName = "InfoLabel_" + IntegerToString(k);
      if(ObjectFind(objName) < 0)
        {
         ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0);
         ObjectSetInteger(0, objName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
        }
      ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, 20 + k * 35);
      ObjectSetString(0, objName, OBJPROP_TEXT, lines[k]);
      ObjectSetString(0, objName, OBJPROP_FONT, "Arial");
      ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, 18);
      ObjectSetInteger(0, objName, OBJPROP_COLOR, txtColor);
      ObjectSetInteger(0, objName, OBJPROP_BACK, false);
     }
   Comment("");
  }

//+------------------------------------------------------------------+
//| Drawing Functions                                                |
//+------------------------------------------------------------------+
void DrawLines()
  {
   ObjectsDeleteAll(0, "Ebat");
   ObjectsDeleteAll(0, "NN");
   ObjectsDeleteAll(0, "Predict");
   for(int i=0; i<trainSize-1; i++)
     {
      double x1 = Time[trainSize - i + predictBars];
      double y1 = rawPrices[i] + offsetY;
      double x2 = Time[trainSize - i + predictBars-1];
      double y2 = rawPrices[i+1] + offsetY;
      int id = ObjectCreate("Ebat" + i, OBJ_TREND, 0, x1, y1, x2, y2);
      if(id == 0) Print(GetLastError());

      ObjectSetString(0, "Ebat" + i, OBJPROP_TEXT, id);
      ObjectSet("Ebat" + i, OBJPROP_COLOR, clrBlue);
      ObjectSet("Ebat" + i, OBJPROP_WIDTH, 5);
      ObjectSet("Ebat" + i, OBJPROP_RAY, 0);
     }
  }

void DrawLinesNN()
  {
   int linesToDraw = trainSize-p;
   double currentY = rawPrices[p - 1];
   for(int i=0; i<linesToDraw; i++)
     {
      double x1 = Time[trainSize - i - p + predictBars + 1];
      double y1 = currentY + offsetY;
      currentY += yValues[i];
      double x2 = Time[trainSize - i - p + predictBars];
      double y2 = currentY + offsetY;

      int id = ObjectCreate("NN" + i, OBJ_TREND, 0, x1, y1, x2, y2);
      if(id == 0) Print("Не удалось создать линию " + GetLastError());

      ObjectSetString(0, "NN" + i, OBJPROP_TEXT, id);
      ObjectSet("NN" + i, OBJPROP_COLOR, clrYellow);
      ObjectSet("NN" + i, OBJPROP_WIDTH, 3);
      ObjectSet("NN" + i, OBJPROP_RAY, 0);
     }
  }

void DrawLinesPredict()
  {
   double currentY = rawPrices[trainSize - 1];
   for(int i=0; i<predictBars; i++)
     {
      double x1 = Time[predictBars - i + 1];
      double y1 = currentY;
      currentY += predictValues[i];
      double x2 = Time[predictBars - i];
      double y2 = currentY;
      
      int id = ObjectCreate("Predict" + i, OBJ_TREND, 0, x1, y1, x2, y2);
      if(id == 0) Print("Не удалось создать линию " + GetLastError());

      ObjectSetString(0, "Predict" + i, OBJPROP_TEXT, id);
      ObjectSet("Predict" + i, OBJPROP_COLOR, clrLimeGreen);
      ObjectSet("Predict" + i, OBJPROP_WIDTH, 3);
      ObjectSet("Predict" + i, OBJPROP_RAY, 0);
     }
  }

void ClearChart()
  {
   for(int i=0; i<trainSize - p; i++)
     {
      string objName = "Ebat" + IntegerToString(i);
      if(ObjectFind(objName) >= 0) ObjectDelete(objName);
     }
  }

//+------------------------------------------------------------------+
//| Math helpers                                                     |
//+------------------------------------------------------------------+
double RandomDouble(double min, double max)
  {
   return min + (max - min) * MathRand() / 32767.0;
  }

double LeakyReLU(double x)
  {
   if(x > 0) return x;
   else return 0.05 * x;
  }

double LeakyReLUDerivative(double y)
  {
   if(y > 0) return 1;
   else return 0.05;
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
