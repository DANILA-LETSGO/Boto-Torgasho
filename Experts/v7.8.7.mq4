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
#define CANDLES_N 3
#define CORR_N 3
#define countHiddenNeuron 6//24
#define trainSize 1536
#define predictBars 24

input double tpMultiplier = 1.0;
input bool readInitData = false;
input int takeProfit = 300;
input int stopLoss = 400;
input int orderThresold = 150;
input bool useDynamicThreshold = false; // Динамический порог входа (ATR 14 / ATR 100)
input bool useDynamicStopLoss = false; // Динамический Stop Loss (ATR 14 / ATR 100)
input bool useDynamicTPLimit = true; // Динамический лимит Тейк-Профита (ATR)
input bool useDynamicTPMultiplier = false; // Динамический множитель Тейк-Профита (ATR)
input bool useAtrAsInput = true; // Подавать ATR на вход нейросети
input bool useMaAsInput = true; // Подавать MA (50 и 200) на вход нейросети
input bool useDayOfWeekAsInput = true; // Подавать день недели (Sin/Cos) на вход нейросети
input bool useCorrelationAsInput = true; // Подавать RSI других пар на вход
input string correlationSymbol1 = "GBPUSD";
input string correlationSymbol2 = "USDJPY";
input bool useCandlesAsInput = true; // Подавать паттерны последних свечей на вход
input bool useTimeAsInput = true; // Подавать циклическое время на вход
input int pass = 30;
input int maxEpochsPerBar = 150; // Максимальное число эпох на один бар (защита от вечного обучения)
int epochsTrainedThisBar = 0;
input int countPredictAnalysys = 6;
input int predictShift = 3;
input double wRange = 0.35;// диапозон случайных значений весов 0.9;
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

double weightsHidden[countHiddenNeuron][p+10+CANDLES_N*3+CORR_N*2];
double thresoldsHidden[countHiddenNeuron];
double weightedSums[countHiddenNeuron];
double outputsHidden[countHiddenNeuron];
double weightsOutputLayer[countHiddenNeuron];
double thresoldOutputLayer;

double vWeightsHidden[countHiddenNeuron][p+10+CANDLES_N*3+CORR_N*2];
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
double errGlobal = 999;

input double tradeErrThresold = 0.235;// Порог ошибки для торговли 0.13
input bool useDynamicError = false; // Динамический порог ошибки (ATR 14 / ATR 100)
double currentTradeErrThreshold = 0.235;
double teachErrThresold = 3.5;
int fastPassCounter;

double predictValues[predictBars];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   currentTradeErrThreshold = tradeErrThresold;
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
         if(k != countHiddenNeuron * (p+10+CANDLES_N*3+CORR_N*2))
           {
            Print("!!!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!! ПИЗДЕЦ !!!!!!!!!!!!");
           }
         else
           {
            int idxRead = 0;
             for(int i=0; i < countHiddenNeuron; i++)
               {
                 for(int j=0; j <= p+9+CANDLES_N*3+CORR_N*2; j++)
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
       // 1. Generate base weights exactly as in v7.8
       for(int i=0; i<countHiddenNeuron; i++)
       {
          for(int j=0; j<=p+1; j++)
            {
               weightsHidden[i][j] = RandomDouble(-wRange, wRange);
            }
          thresoldsHidden[i] = RandomDouble(-wRange, wRange);
          weightsOutputLayer[i] = RandomDouble(-wRange, wRange);
       }
       thresoldOutputLayer = RandomDouble(-wRange, wRange);
       
       // 2. Generate extra weights from the tail of generator
       for(int i=0; i<countHiddenNeuron; i++)
       {
          for(int j=p+2; j<=p+9+CANDLES_N*3+CORR_N*2; j++)
            {
               weightsHidden[i][j] = RandomDouble(-wRange, wRange);
            }
       }
       
       // 3. Build opta string
       //string opta = "";
       //for(int i=0; i<countHiddenNeuron; i++)
       //{
          //for(int j=0; j<=p+9+CANDLES_N*3+CORR_N*2; j++)
            //{
               //opta += DoubleToStr(weightsHidden[i][j], 8) + "|=|";
            //}
       //}
       //WriteToFile(opta, "Exp7_weightsHidden.txt");
   
       //SaveThresoldsHidden(false);
       //SaveWeightsOutputLayer(false);
       Print("======================================= Веса Инициализированы ===================================");
     }
     
   for(int i=0; i<countHiddenNeuron; i++)
     {
      for(int j=0; j<=p+9+CANDLES_N*3+CORR_N*2; j++)
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
      for(int j=0; j<=p+9+CANDLES_N*3+CORR_N*2; j++)
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
      
      int shift = trainSize - (iSample + p - 1);
      double atrInput = 0;
      double atrInput2 = 0;
      if(useAtrAsInput)
        {
         double atr = iATR(NULL, 0, 14, shift);
         atrInput = atr / range;
         double atr2 = iATR(NULL, 0, 100, shift);
         atrInput2 = atr2 / range;
        }
        
      double maInput = 0;
      double maInput2 = 0;
      if(useMaAsInput)
        {
         double closePrice = iClose(NULL, 0, shift);
         double ma50 = iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE, shift);
         double ma200 = iMA(NULL, 0, 200, 0, MODE_SMA, PRICE_CLOSE, shift);
         maInput = 2.0 * MathArctan((closePrice - ma50) / range) / 3.1415926535;
         maInput2 = 2.0 * MathArctan((closePrice - ma200) / range) / 3.1415926535;
        }
        
      double candleInputs[99];
      for(int c=0; c<CANDLES_N*3; c++) candleInputs[c] = 0;
      if(useCandlesAsInput)
        {
         for(int c=0; c<CANDLES_N; c++)
           {
            int cShift = shift + c;
            double cOpen = iOpen(NULL, 0, cShift);
            double cHigh = iHigh(NULL, 0, cShift);
            double cLow = iLow(NULL, 0, cShift);
            double cClose = iClose(NULL, 0, cShift);
            
            candleInputs[c*3] = (cClose - cOpen) / range;
            candleInputs[c*3+1] = (cHigh - MathMax(cOpen, cClose)) / range;
            candleInputs[c*3+2] = (MathMin(cOpen, cClose) - cLow) / range;
           }
        }
        
      double corr1Inputs[99];
      double corr2Inputs[99];
      for(int c=0; c<CORR_N; c++) { corr1Inputs[c] = 0; corr2Inputs[c] = 0; }
      if(useCorrelationAsInput)
        {
         for(int c=0; c<CORR_N; c++)
           {
            int cShift = shift + c;
            double rsi1 = iRSI(correlationSymbol1, 0, 14, PRICE_CLOSE, cShift);
            double rsi2 = iRSI(correlationSymbol2, 0, 14, PRICE_CLOSE, cShift);
            corr1Inputs[c] = (rsi1 - 50.0) / 100.0;
            corr2Inputs[c] = (rsi2 - 50.0) / 100.0;
           }
        }
        
      double sinTime = 0;
      double cosTime = 0;
      double sinDay = 0;
      double cosDay = 0;
      if(useTimeAsInput || useDayOfWeekAsInput)
        {
         datetime barTime = iTime(NULL, 0, shift);
         if(useTimeAsInput)
           {
            double t = (TimeHour(barTime) * 60 + TimeMinute(barTime)) / 1440.0;
            sinTime = MathSin(2.0 * 3.1415926535 * t);
            cosTime = MathCos(2.0 * 3.1415926535 * t);
           }
         if(useDayOfWeekAsInput)
           {
            int day = TimeDayOfWeek(barTime);
            if(day == 0) day = 1;
            if(day == 6) day = 5;
            double tDay = (day - 1) / 5.0;
            sinDay = MathSin(2.0 * 3.1415926535 * tDay);
            cosDay = MathCos(2.0 * 3.1415926535 * tDay);
           }
        }

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
         wSum += weightsHidden[i][p+2] * atrInput;
         wSum += weightsHidden[i][p+3] * atrInput2;
         wSum += weightsHidden[i][p+4] * maInput;
         wSum += weightsHidden[i][p+5] * maInput2;
         for(int c=0; c<CANDLES_N*3; c++)
           {
            wSum += weightsHidden[i][p+6+c] * candleInputs[c];
           }
         wSum += weightsHidden[i][p+6+CANDLES_N*3] * sinTime;
         wSum += weightsHidden[i][p+7+CANDLES_N*3] * cosTime;
         wSum += weightsHidden[i][p+8+CANDLES_N*3] * sinDay;
         wSum += weightsHidden[i][p+9+CANDLES_N*3] * cosDay;
         for(int c=0; c<CORR_N; c++)
           {
            wSum += weightsHidden[i][p+10+CANDLES_N*3+c] * corr1Inputs[c];
            wSum += weightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] * corr2Inputs[c];
           }
         
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
      
      // Считаем чистую нормализованную ошибку (без привязки к размеру свечей)
      errGlobal += errOutput * errOutput;
     }
   // Переводим в удобный масштаб (например, умножаем на 100, чтобы значения были около 0.1 - 1.0)
   errGlobal = (errGlobal / (trainSize - p)) * 100;
  }

//+------------------------------------------------------------------+
//| Train with Rolling Window                                        |
//+------------------------------------------------------------------+
void Train()
  {
   
   if(allowSkipTrain == true && errGlobal < currentTradeErrThreshold && fastPassCounter < skipTrainMaxPasses)
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
         
         int shift = trainSize - (iSample + p - 1);
         double atrInput = 0;
         double atrInput2 = 0;
         if(useAtrAsInput)
           {
            double atr = iATR(NULL, 0, 14, shift);
            atrInput = atr / range;
            double atr2 = iATR(NULL, 0, 100, shift);
            atrInput2 = atr2 / range;
           }
           
         double maInput = 0;
         double maInput2 = 0;
         if(useMaAsInput)
           {
            double closePrice = iClose(NULL, 0, shift);
            double ma50 = iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE, shift);
            double ma200 = iMA(NULL, 0, 200, 0, MODE_SMA, PRICE_CLOSE, shift);
            maInput = 2.0 * MathArctan((closePrice - ma50) / range) / 3.1415926535;
            maInput2 = 2.0 * MathArctan((closePrice - ma200) / range) / 3.1415926535;
           }
           
         double candleInputs[99];
         for(int c=0; c<CANDLES_N*3; c++) candleInputs[c] = 0;
         if(useCandlesAsInput)
           {
            for(int c=0; c<CANDLES_N; c++)
              {
               int cShift = shift + c;
               double cOpen = iOpen(NULL, 0, cShift);
               double cHigh = iHigh(NULL, 0, cShift);
               double cLow = iLow(NULL, 0, cShift);
               double cClose = iClose(NULL, 0, cShift);
               
               candleInputs[c*3] = (cClose - cOpen) / range;
               candleInputs[c*3+1] = (cHigh - MathMax(cOpen, cClose)) / range;
               candleInputs[c*3+2] = (MathMin(cOpen, cClose) - cLow) / range;
              }
           }
            
          double corr1Inputs[99];
          double corr2Inputs[99];
          for(int c=0; c<CORR_N; c++) { corr1Inputs[c] = 0; corr2Inputs[c] = 0; }
          if(useCorrelationAsInput)
            {
             for(int c=0; c<CORR_N; c++)
               {
                int cShift = shift + c;
                double rsi1 = iRSI(correlationSymbol1, 0, 14, PRICE_CLOSE, cShift);
                double rsi2 = iRSI(correlationSymbol2, 0, 14, PRICE_CLOSE, cShift);
                corr1Inputs[c] = (rsi1 - 50.0) / 100.0;
                corr2Inputs[c] = (rsi2 - 50.0) / 100.0;
               }
            }
           
         double sinTime = 0;
         double cosTime = 0;
         double sinDay = 0;
         double cosDay = 0;
         if(useTimeAsInput || useDayOfWeekAsInput)
           {
            datetime barTime = iTime(NULL, 0, shift);
            if(useTimeAsInput)
              {
               double t = (TimeHour(barTime) * 60 + TimeMinute(barTime)) / 1440.0;
               sinTime = MathSin(2.0 * 3.1415926535 * t);
               cosTime = MathCos(2.0 * 3.1415926535 * t);
              }
            if(useDayOfWeekAsInput)
              {
               int day = TimeDayOfWeek(barTime);
               if(day == 0) day = 1;
               if(day == 6) day = 5;
               double tDay = (day - 1) / 5.0;
               sinDay = MathSin(2.0 * 3.1415926535 * tDay);
               cosDay = MathCos(2.0 * 3.1415926535 * tDay);
              }
           }

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
            wSum += weightsHidden[i][p+2] * atrInput;
            wSum += weightsHidden[i][p+3] * atrInput2;
            wSum += weightsHidden[i][p+4] * maInput;
            wSum += weightsHidden[i][p+5] * maInput2;
            for(int c=0; c<CANDLES_N*3; c++)
              {
               wSum += weightsHidden[i][p+6+c] * candleInputs[c];
              }
            wSum += weightsHidden[i][p+6+CANDLES_N*3] * sinTime;
            wSum += weightsHidden[i][p+7+CANDLES_N*3] * cosTime;
            wSum += weightsHidden[i][p+8+CANDLES_N*3] * sinDay;
            wSum += weightsHidden[i][p+9+CANDLES_N*3] * cosDay;
            for(int c=0; c<CORR_N; c++)
              {
               wSum += weightsHidden[i][p+10+CANDLES_N*3+c] * corr1Inputs[c];
               wSum += weightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] * corr2Inputs[c];
              }
            
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
         // Считаем чистую нормализованную ошибку
         errGlobal += errOutput * errOutput;
         
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
              
              double gradAtr = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * atrInput;
              vWeightsHidden[i][p+2] = momentum * vWeightsHidden[i][p+2] - a * gradAtr;
              weightsHidden[i][p+2] += vWeightsHidden[i][p+2];
              
              double gradAtr2 = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * atrInput2;
              vWeightsHidden[i][p+3] = momentum * vWeightsHidden[i][p+3] - a * gradAtr2;
              weightsHidden[i][p+3] += vWeightsHidden[i][p+3];
              
              double gradMa = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * maInput;
              vWeightsHidden[i][p+4] = momentum * vWeightsHidden[i][p+4] - a * gradMa;
              weightsHidden[i][p+4] += vWeightsHidden[i][p+4];
              
              double gradMa2 = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * maInput2;
              vWeightsHidden[i][p+5] = momentum * vWeightsHidden[i][p+5] - a * gradMa2;
              weightsHidden[i][p+5] += vWeightsHidden[i][p+5];
              
              for(int c=0; c<CANDLES_N*3; c++)
                {
                 double gradCandle = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * candleInputs[c];
                 vWeightsHidden[i][p+6+c] = momentum * vWeightsHidden[i][p+6+c] - a * gradCandle;
                 weightsHidden[i][p+6+c] += vWeightsHidden[i][p+6+c];
                }
               
               double gradSin = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * sinTime;
               vWeightsHidden[i][p+6+CANDLES_N*3] = momentum * vWeightsHidden[i][p+6+CANDLES_N*3] - a * gradSin;
               weightsHidden[i][p+6+CANDLES_N*3] += vWeightsHidden[i][p+6+CANDLES_N*3];
               
               double gradCos = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * cosTime;
               vWeightsHidden[i][p+7+CANDLES_N*3] = momentum * vWeightsHidden[i][p+7+CANDLES_N*3] - a * gradCos;
               weightsHidden[i][p+7+CANDLES_N*3] += vWeightsHidden[i][p+7+CANDLES_N*3];
               
               double gradSinDay = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * sinDay;
               vWeightsHidden[i][p+8+CANDLES_N*3] = momentum * vWeightsHidden[i][p+8+CANDLES_N*3] - a * gradSinDay;
               weightsHidden[i][p+8+CANDLES_N*3] += vWeightsHidden[i][p+8+CANDLES_N*3];
               
               double gradCosDay = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * cosDay;
               vWeightsHidden[i][p+9+CANDLES_N*3] = momentum * vWeightsHidden[i][p+9+CANDLES_N*3] - a * gradCosDay;
               weightsHidden[i][p+9+CANDLES_N*3] += vWeightsHidden[i][p+9+CANDLES_N*3];
               
               for(int c=0; c<CORR_N; c++)
                 {
                  double gradCorr1 = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * corr1Inputs[c];
                  vWeightsHidden[i][p+10+CANDLES_N*3+c] = momentum * vWeightsHidden[i][p+10+CANDLES_N*3+c] - a * gradCorr1;
                  weightsHidden[i][p+10+CANDLES_N*3+c] += vWeightsHidden[i][p+10+CANDLES_N*3+c];
                  
                  double gradCorr2 = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * corr2Inputs[c];
                  vWeightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] = momentum * vWeightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] - a * gradCorr2;
                  weightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] += vWeightsHidden[i][p+10+CANDLES_N*3+CORR_N+c];
                 }
               
               double gradThresoldHid = -errHiden[i] * LeakyReLUDerivative(weightedSums[i]);
              vThresoldsHidden[i] = momentum * vThresoldsHidden[i] - a * gradThresoldHid;
              thresoldsHidden[i] += vThresoldsHidden[i];
           }
        }
      errGlobal = (errGlobal / (trainSize - p)) * 100;
     }

   Print("Итерация " + countTeaches + " === Ошибко " + errGlobal + " Save Timer " + saveTimer);
 
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
      
      double atrInput = 0;
      double atrInput2 = 0;
      if(useAtrAsInput)
        {
         int shift = MathMax(1, 1 - pre);
         double atr = iATR(NULL, 0, 14, shift);
         atrInput = atr / range;
         double atr2 = iATR(NULL, 0, 100, shift);
         atrInput2 = atr2 / range;
        }
        
      double maInput = 0;
      double maInput2 = 0;
      if(useMaAsInput)
        {
         int shift = MathMax(1, 1 - pre);
         double closePrice = iClose(NULL, 0, shift);
         double ma50 = iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE, shift);
         double ma200 = iMA(NULL, 0, 200, 0, MODE_SMA, PRICE_CLOSE, shift);
         maInput = 2.0 * MathArctan((closePrice - ma50) / range) / 3.1415926535;
         maInput2 = 2.0 * MathArctan((closePrice - ma200) / range) / 3.1415926535;
        }
        
      double candleInputs[99];
      for(int c=0; c<CANDLES_N*3; c++) candleInputs[c] = 0;
      if(useCandlesAsInput)
        {
         for(int c=0; c<CANDLES_N; c++)
           {
            int cShift = MathMax(1, 1 - pre + c);
            double cOpen = iOpen(NULL, 0, cShift);
            double cHigh = iHigh(NULL, 0, cShift);
            double cLow = iLow(NULL, 0, cShift);
            double cClose = iClose(NULL, 0, cShift);
            
            candleInputs[c*3] = (cClose - cOpen) / range;
            candleInputs[c*3+1] = (cHigh - MathMax(cOpen, cClose)) / range;
            candleInputs[c*3+2] = (MathMin(cOpen, cClose) - cLow) / range;
           }
        }
         
       double corr1Inputs[99];
       double corr2Inputs[99];
       for(int c=0; c<CORR_N; c++) { corr1Inputs[c] = 0; corr2Inputs[c] = 0; }
       if(useCorrelationAsInput)
         {
          for(int c=0; c<CORR_N; c++)
            {
             int cShift = MathMax(1, 1 - pre + c);
             double rsi1 = iRSI(correlationSymbol1, 0, 14, PRICE_CLOSE, cShift);
             double rsi2 = iRSI(correlationSymbol2, 0, 14, PRICE_CLOSE, cShift);
             corr1Inputs[c] = (rsi1 - 50.0) / 100.0;
             corr2Inputs[c] = (rsi2 - 50.0) / 100.0;
            }
         }
        
      double sinTime = 0;
      double cosTime = 0;
      double sinDay = 0;
      double cosDay = 0;
      if(useTimeAsInput || useDayOfWeekAsInput)
        {
         datetime barTime = iTime(NULL, 0, 1) + pre * PeriodSeconds();
         if(useTimeAsInput)
           {
            double t = (TimeHour(barTime) * 60 + TimeMinute(barTime)) / 1440.0;
            sinTime = MathSin(2.0 * 3.1415926535 * t);
            cosTime = MathCos(2.0 * 3.1415926535 * t);
           }
         if(useDayOfWeekAsInput)
           {
            int day = TimeDayOfWeek(barTime);
            if(day == 0) day = 1;
            if(day == 6) day = 5;
            double tDay = (day - 1) / 5.0;
            sinDay = MathSin(2.0 * 3.1415926535 * tDay);
            cosDay = MathCos(2.0 * 3.1415926535 * tDay);
           }
        }

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
         wSum += weightsHidden[i][p+2] * atrInput;
         wSum += weightsHidden[i][p+3] * atrInput2;
         wSum += weightsHidden[i][p+4] * maInput;
         wSum += weightsHidden[i][p+5] * maInput2;
         for(int c=0; c<CANDLES_N*3; c++)
           {
            wSum += weightsHidden[i][p+6+c] * candleInputs[c];
           }
         wSum += weightsHidden[i][p+6+CANDLES_N*3] * sinTime;
         wSum += weightsHidden[i][p+7+CANDLES_N*3] * cosTime;
         wSum += weightsHidden[i][p+8+CANDLES_N*3] * sinDay;
         wSum += weightsHidden[i][p+9+CANDLES_N*3] * cosDay;
         for(int c=0; c<CORR_N; c++)
           {
            wSum += weightsHidden[i][p+10+CANDLES_N*3+c] * corr1Inputs[c];
            wSum += weightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] * corr2Inputs[c];
           }
         
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

   if(errGlobal < currentTradeErrThreshold)
     {
         Trade();
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
   double predictedPrice = rawPrices[trainSize - 1]; // Начинаем от текущей цены закрытия

   // 1. Проматываем "эхо" (запаздывание)
   for(int j=0; j<predictShift; j++)
     {
      if(j < predictBars) predictedPrice += predictValues[j];
     }
     
   // 2. Фиксируем опорную точку (baseline) ПОСЛЕ запаздывания!
   double baselinePrice = predictedPrice; 

   for(int i=0; i<total; i++)
     {
      int idx = i + predictShift;
      if(idx >= predictBars) break;
      predictedPrice += predictValues[idx]; // Добавляем следующий шаг прогноза
      
      // Сравниваем прогноз с ОПОРНОЙ точкой, а не с реальным рынком
      if(predictedPrice > baselinePrice)
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
      pipsHigh = (averageHigh - baselinePrice) / _Point; // Считаем потенциал от опорной точки
     }
   if(countLow > 0)
     {
      double averageLow = sumLow / (double)countLow;
      pipsLow = (baselinePrice - averageLow) / _Point; // Считаем потенциал от опорной точки
     }

   double dynamicThreshold = orderThresold;
   double dynamicStopLossLimit = stopLoss;
   double dynamicTakeProfitLimit = takeProfit;
   double dynamicTpMultiplier = tpMultiplier;
   
   if(useDynamicThreshold || useDynamicStopLoss || useDynamicTPLimit || useDynamicTPMultiplier)
     {
      double atrFast = iATR(_Symbol, 0, 14, 1);
      double atrSlow = iATR(_Symbol, 0, 100, 1);
      if(atrSlow > 0) 
        {
         double volRatio = atrFast / atrSlow;
         if(volRatio < 0.5) volRatio = 0.5;
         if(volRatio > 2.0) volRatio = 2.0;
         
         if(useDynamicThreshold) dynamicThreshold = orderThresold * volRatio;
         if(useDynamicStopLoss) dynamicStopLossLimit = stopLoss * volRatio;
         if(useDynamicTPLimit) dynamicTakeProfitLimit = takeProfit * volRatio;
         if(useDynamicTPMultiplier) dynamicTpMultiplier = tpMultiplier * volRatio;
        }
     }
     
   bool isBuySignal = (pipsHigh - pipsLow > dynamicThreshold);
   bool isSellSignal = (pipsLow - pipsHigh > dynamicThreshold);

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
      int t = pipsHigh > dynamicTakeProfitLimit ? (int)dynamicTakeProfitLimit : (int)(pipsHigh * dynamicTpMultiplier);
      int s = pipsLow < dynamicStopLossLimit ? (int)dynamicStopLossLimit : pipsLow;
      double sl = Bid - s * _Point;
      double tp = Bid + t *_Point;
      ticket = OrderSend(_Symbol, OP_BUY, 0.1, Ask, 5, sl, tp, "", magicNumber);
     }
   else if(isSellSignal)
     {
      int t = pipsLow > dynamicTakeProfitLimit ? (int)dynamicTakeProfitLimit : (int)(pipsLow * dynamicTpMultiplier);
      int s = pipsHigh < dynamicStopLossLimit ? (int)dynamicStopLossLimit : pipsHigh;
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
   double totalProfitPips = 0;
   int elementsFound = 0;
   
   // 1. Собираем свежие сделки в массив для сортировки
   // Массив формата [количество_сделок][2], где:
   // [i][0] - время закрытия (для сортировки)
   // [i][1] - тикет ордера
   double recentTrades[][2];
   int tradesCount = 0;
   int safetyBuffer = 0; // Счётчик старых сделок подряд
   
   for(int i = historyTotal - 1; i >= 0; i--)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
        {
         if(OrderSymbol() == _Symbol && OrderMagicNumber() == magicNumber)
           {
            int type = OrderType();
            if(type == OP_BUY || type == OP_SELL)
              {
               if(OrderCloseTime() > lastResetTime)
                 {
                  safetyBuffer = 0; // Сбрасываем буфер, если нашли свежую сделку
                  
                  // Расширяем массив порциями для оптимизации
                  if(tradesCount >= ArrayRange(recentTrades, 0))
                     ArrayResize(recentTrades, tradesCount + 50);
                     
                  recentTrades[tradesCount][0] = (double)OrderCloseTime() * 1000000.0 + (double)(1000000 - i);
                  recentTrades[tradesCount][1] = (double)OrderTicket();
                  tradesCount++;
                 }
               else
                 {
                  safetyBuffer++;
                  // Если встретили 30 старых сделок подряд, считаем, что дальше только старые
                  if(safetyBuffer >= 30) break;
                 }
              }
           }
        }
     }
     
   if(tradesCount == 0) return;
   
   // Обрезаем массив до реального размера и сортируем по времени по убыванию (MODE_DESCEND)
   ArrayResize(recentTrades, tradesCount);
   ArraySort(recentTrades, WHOLE_ARRAY, 0, MODE_DESCEND);
   
   // 2. Обрабатываем отсортированные сделки
   if (onlyOneOrder)
     {
      for(int i = 0; i < tradesCount; i++)
        {
         if(OrderSelect((int)recentTrades[i][1], SELECT_BY_TICKET, MODE_HISTORY))
           {
            int type = OrderType();
            double profitPips = (type == OP_BUY) ? (OrderClosePrice() - OrderOpenPrice()) / _Point : (OrderOpenPrice() - OrderClosePrice()) / _Point;
            
            totalProfitPips += profitPips;
            elementsFound++;
            
            if(elementsFound >= n) break;
           }
        }
     }
   else
     {
      int currentType = -1;
      double basketSum = 0;
      int basketCount = 0;
      
      for(int i = 0; i < tradesCount; i++)
        {
         if(OrderSelect((int)recentTrades[i][1], SELECT_BY_TICKET, MODE_HISTORY))
           {
            int type = OrderType();
            double profitPips = (type == OP_BUY) ? (OrderClosePrice() - OrderOpenPrice()) / _Point : (OrderOpenPrice() - OrderClosePrice()) / _Point;
            
            if(currentType == -1) currentType = type;
            
            if(type == currentType)
              {
               basketSum += profitPips;
               basketCount++;
              }
            else
              {
               totalProfitPips += (basketSum / basketCount);
               elementsFound++;
               
               if (elementsFound >= n) break;
               
               currentType = type;
               basketSum = profitPips;
               basketCount = 1;
              }
           }
        }
        
      if (basketCount > 0 && elementsFound < n)
        {
         totalProfitPips += (basketSum / basketCount);
         elementsFound++;
        }
     }
     
   if(elementsFound > 0 && totalProfitPips <= -maxLossThresholdPips)
     {
      if (onlyOneOrder)
         Print("Суммарный убыток последних " + IntegerToString(elementsFound) + " сделок составил " + DoubleToString(-totalProfitPips, 1) + " пунктов. Сброс весов!");
      else
         Print("Суммарный убыток последних " + IntegerToString(elementsFound) + " КОРЗИН составил " + DoubleToString(-totalProfitPips, 1) + " пунктов. Сброс весов!");
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
        
      // Обновляем время ПОСЛЕ закрытия сделок, чтобы проигнорировать их при следующей проверке
      lastResetTime = TimeCurrent();
     }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   static int lastDay = -1;
   if(lastDay == -1) lastDay = TimeDay(Time[0]);
   
   if(TimeDay(Time[0]) != lastDay)
     {
      Print("==================================================================");
      Print("Начало нового дня! Сбрасываем веса (Daily Rolling Window)...");
      InitWeights();
      lastResetTime = TimeCurrent();
      lastDay = TimeDay(Time[0]);
     }

   currentTradeErrThreshold = tradeErrThresold;
   if(useDynamicError)
     {
      double atrFast = iATR(_Symbol, 0, 14, 1);
      double atrSlow = iATR(_Symbol, 0, 100, 1);
      if(atrSlow > 0)
        {
         double volRatio = atrFast / atrSlow;
         if(volRatio < 0.5) volRatio = 0.5;
         if(volRatio > 2.0) volRatio = 2.0;
         currentTradeErrThreshold = tradeErrThresold * volRatio;
        }
     }

   CheckDrawdownAndReset();

   if(IsNewBar())
     {
      ebobo = true;
      epochsTrainedThisBar = 0; // Сбрасываем счетчик при появлении нового бара
      InitBars();
      if(allowSkipTrain) CalculateError();
      fastPassCounter++;
     }

   if(ebobo)
     {
      Train();
      epochsTrainedThisBar += pass; // Учитываем пройденные эпохи
      
      Predict();
      DrawLines();
      DrawLinesNN();
      DrawLinesPredict();
      
      // Если лимит исчерпан, а бот так и не открыл сделку (ebobo все еще true)
      if(ebobo && epochsTrainedThisBar >= maxEpochsPerBar)
        {
         ebobo = false; // Принудительно завершаем обучение на текущем баре
        }
     }

   string status = (errGlobal < currentTradeErrThreshold) ? "[OK] READY (Trading)" : "[!] LEARNING";
   color txtColor = (color)ChartGetInteger(0, CHART_COLOR_FOREGROUND);
   ChartSetInteger(0, CHART_FOREGROUND, false);
   string lines[13];
   lines[0] = "--- Antigravity NN v7 ---";
   lines[1] = "Status: " + status;
   lines[2] = "Arch: Inp(" + IntegerToString(p+10+CANDLES_N*3+CORR_N*2) + ") -> Hid(" + IntegerToString(countHiddenNeuron) + ") -> Out(Linear)";
   lines[3] = "Mem (trainSize): " + IntegerToString(trainSize) + " bars | RSI: Enabled (Dynamic)";
   lines[4] = "Global Error: " + DoubleToString(errGlobal, 5) + " / " + DoubleToString(currentTradeErrThreshold, 5) + (useDynamicError ? " (Dyn)" : "");
   lines[5] = "Passes Done: " + IntegerToString(countTeaches);
   lines[6] = "Pass (Epochs per tick): " + IntegerToString(pass);
   lines[7] = "Learning Rate (aStep): " + DoubleToString(aStep, 6);
   lines[8] = "--- Trading Logic ---";
   lines[9] = "Trade Horizon: " + IntegerToString(countPredictAnalysys) + " (Vis: " + IntegerToString(predictBars) + ") | Shift: " + IntegerToString(predictShift);
   
   double displayThreshold = orderThresold;
   if(useDynamicThreshold)
     {
      double atrFast = iATR(_Symbol, 0, 14, 1);
      double atrSlow = iATR(_Symbol, 0, 100, 1);
      if(atrSlow > 0) 
        {
         double volRatio = atrFast / atrSlow;
         if(volRatio < 0.5) volRatio = 0.5;
         if(volRatio > 2.0) volRatio = 2.0;
         displayThreshold = orderThresold * volRatio;
        }
      lines[10] = "Order Threshold: " + DoubleToString(displayThreshold, 1) + " (Dynamic)";
     }
   else
     {
      lines[10] = "Order Threshold: " + IntegerToString(orderThresold) + " points";
     }
     
   if(useDynamicStopLoss)
     {
      double atrFast = iATR(_Symbol, 0, 14, 1);
      double atrSlow = iATR(_Symbol, 0, 100, 1);
      double displaySL = stopLoss;
      if(atrSlow > 0) 
        {
         double volRatio = atrFast / atrSlow;
         if(volRatio < 0.5) volRatio = 0.5;
         if(volRatio > 2.0) volRatio = 2.0;
         displaySL = stopLoss * volRatio;
        }
      lines[11] = "Stop Loss: " + DoubleToString(displaySL, 1) + " (Dynamic)";
     }
   else
     {
      lines[11] = "Stop Loss: " + IntegerToString(stopLoss) + " points";
     }
     
   double displayTP = takeProfit;
   double displayMult = tpMultiplier;
   if(useDynamicTPLimit || useDynamicTPMultiplier)
     {
      double atrFast = iATR(_Symbol, 0, 14, 1);
      double atrSlow = iATR(_Symbol, 0, 100, 1);
      if(atrSlow > 0) 
        {
         double volRatio = atrFast / atrSlow;
         if(volRatio < 0.5) volRatio = 0.5;
         if(volRatio > 2.0) volRatio = 2.0;
         if(useDynamicTPLimit) displayTP = takeProfit * volRatio;
         if(useDynamicTPMultiplier) displayMult = tpMultiplier * volRatio;
        }
     }
   
   string dynStr = (useDynamicTPLimit || useDynamicTPMultiplier) ? " (Dyn ATR)" : "";
   lines[12] = "TP Limit: " + DoubleToString(displayTP, 1) + " | Mult: " + DoubleToString(displayMult, 2) + dynStr;
   
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
