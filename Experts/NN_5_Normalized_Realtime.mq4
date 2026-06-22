//+------------------------------------------------------------------+
//|                                   NN_5_Normalized_Realtime.mq4   |
//|                             Copyright 2026, Antigravity AI Team. |
//|                                             https://antigravity  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Antigravity AI Team."
#property link      "https://antigravity"
#property version   "1.10"
#property description "Normalized Multi-Layer Perceptron (NN) Real-time Scalper"
#property strict

#define p 288
#define countHiddenNeuron 8
#define trainSize 512
#define predictBars 49

input double tpMultiplier = 1.0;
input bool readInitData = false;
input int takeProfit = 100;
input int stopLoss = 150;
input int orderThresold = 80;
input int pass = 30;
int maxTeach = 1000;
input int countPredictAnalysys = 6;
input int predictShift = 3;

input double wRange = 0.28; // weights initialization range
input bool isRealtime = false;
input bool saveLearningProgress = false;
input bool ovverideSaveFile = false;
input int saveTimerThresold = 100;
input bool allowSkipTrain = false;
input double aStep = 0.01;

// Neural Network arrays
double weightsHidden[countHiddenNeuron][p];
double thresoldsHidden[countHiddenNeuron];
double weightedSums[countHiddenNeuron];
double outputsHidden[countHiddenNeuron];
double weightsOutputLayer[countHiddenNeuron];
double thresoldOutputLayer;
double yValues[trainSize - p];
double etalons[trainSize];

// Normalization variables
double minPrice = 0.0;
double maxPrice = 0.0;

double offsetY = 0.0;
int countTeaches = 0;
double errIncrease = 0;

int saveTimer = 0;
int ticket;
bool trainRequired = true;
bool nonTeacheble = false;
double errGlobal = 999.0;
int countTryed = 0;
input double tradeErrThresold = 0.235; // Error threshold for trading
double teachErrThresold = 3.5;
int fastPassCounter = 0;

double predictValues[predictBars];

// Forward declarations
void InitBars();
void InitWeights();
void Train();
void Predict();
void Trade();
void DrawLines();
void DrawLinesNN();
void DrawLinesPredict();
void ClearChart();
double RandomDouble(double minVal, double maxVal);
double Sigmoid(double x);
double SigmoidDerivative(double y);
bool IsNewBar();
void SaveWeightsData(bool isLearning);
void SaveWeightsHiddenLayer(bool isLearning);
void SaveThresoldsHidden(bool isLearning);
void SaveWeightsOutputLayer(bool isLearning);
void WriteToFile(string data, string fileName);

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   ObjectsDeleteAll();
   InitBars();
   InitWeights();
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ClearChart();
   ObjectsDeleteAll();
}

//+------------------------------------------------------------------+
//| Initialize network weights                                       |
//+------------------------------------------------------------------+
void InitWeights()
{
   countTeaches = 0;
   errGlobal = 999.0;

   if(readInitData)
   {
      string strWeightsHidden;
      int handle = FileOpen("weightsHidden.txt", FILE_TXT|FILE_READ);
      if(handle > 0)
      {
         strWeightsHidden = FileReadString(handle);
         string sep = "=";
         ushort u_sep = StringGetCharacter(sep, 0);
         string result[];
         int k = StringSplit(strWeightsHidden, u_sep, result) - 1;
         
         if(k != countHiddenNeuron * p)
         {
            Print("Error: weightsHidden.txt file size mismatch.");
         }
         else
         {
            int idxRead = 0;
            for(int i=0; i < countHiddenNeuron; i++)
            {
               for(int j=0; j < p; j++)
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
         Print("Warning: weightsHidden.txt could not be loaded. Error: ", GetLastError());
      }
      
      string strThresoldsHidden;
      handle = FileOpen("thresoldsHidden.txt", FILE_TXT|FILE_READ);
      if(handle > 0)
      {
         strThresoldsHidden = FileReadString(handle);
         string result[];
         int k = StringSplit(strThresoldsHidden, StringGetCharacter("=", 0), result) - 1;
         
         if(k != countHiddenNeuron)
         {
            Print("Error: thresoldsHidden.txt file size mismatch.");
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
      
      string strWeightsOutputLayer;
      handle = FileOpen("weightsOutputLayer.txt", FILE_TXT|FILE_READ);
      if(handle > 0)
      {
         strWeightsOutputLayer = FileReadString(handle);
         string result[];
         int k = StringSplit(strWeightsOutputLayer, StringGetCharacter("=", 0), result) - 1;
         
         if(k != countHiddenNeuron)
         {
            Print("Error: weightsOutputLayer.txt file size mismatch.");
         }
         else
         {
            int idxRead = 0;
            for(int i=0; i<countHiddenNeuron; i++)
            {
               weightsOutputLayer[i] = StrToDouble(result[idxRead]);
               idxRead++;
            }
         }
         FileClose(handle);
      }
   }
   else
   {
      string opta = "";
      for(int i=0; i<countHiddenNeuron; i++)
      {
         for(int j=0; j<p; j++)
         {
            weightsHidden[i][j] = RandomDouble(-wRange, wRange);
            opta += DoubleToString(weightsHidden[i][j], 8) + "|=|";
         }
         thresoldsHidden[i] = RandomDouble(-wRange, wRange);
         weightsOutputLayer[i] = RandomDouble(-wRange, wRange);
      }

      WriteToFile(opta, "weightsHidden.txt");
      SaveThresoldsHidden(false);
      SaveWeightsOutputLayer(false);
      Print("Weights initialized successfully.");
   }
}

//+------------------------------------------------------------------+
//| Save all weights files                                           |
//+------------------------------------------------------------------+
void SaveWeightsData(bool isLearning)
{
   SaveWeightsHiddenLayer(isLearning);
   SaveThresoldsHidden(isLearning);
   SaveWeightsOutputLayer(isLearning);
}

//+------------------------------------------------------------------+
//| Save hidden layer weights                                        |
//+------------------------------------------------------------------+
void SaveWeightsHiddenLayer(bool isLearning)
{
   string data = "";
   for(int i=0; i<countHiddenNeuron; i++)
   {
      for(int j=0; j<p; j++)
      {
         data += DoubleToString(weightsHidden[i][j], 8) + "|=|";
      }
   }
   if(isLearning)
      WriteToFile(data, "L_weightsHidden.txt");
   else
      WriteToFile(data, "weightsHidden.txt");
}

//+------------------------------------------------------------------+
//| Save hidden layer thresholds                                     |
//+------------------------------------------------------------------+
void SaveThresoldsHidden(bool isLearning)
{
   string data = "";
   for(int i=0; i<countHiddenNeuron; i++)
   {
      data += DoubleToString(thresoldsHidden[i], 8) + "|=|";
   }
   if(isLearning)
      WriteToFile(data, "L_thresoldsHidden.txt");
   else
      WriteToFile(data, "thresoldsHidden.txt");
}

//+------------------------------------------------------------------+
//| Save output layer weights                                        |
//+------------------------------------------------------------------+
void SaveWeightsOutputLayer(bool isLearning)
{
   string data = "";
   for(int i=0; i<countHiddenNeuron; i++)
   {
      data += DoubleToString(weightsOutputLayer[i], 8) + "|=|";
   }
   if(isLearning)
      WriteToFile(data, "L_weightsOutputLayer.txt");
   else
      WriteToFile(data, "weightsOutputLayer.txt");
}

//+------------------------------------------------------------------+
//| Helper to write string to file                                   |
//+------------------------------------------------------------------+
void WriteToFile(string data, string fileName)
{
   FileDelete(fileName);
   int strLength = StringLen(data);
   int handle = FileOpen(fileName, FILE_TXT|FILE_READ|FILE_WRITE, ';');
   if(handle < 1)
   {
      Print("Error: Failed to write to ", fileName, ". Error code: ", GetLastError());
   }
   else
   {
      FileWriteString(handle, data, strLength);
      FileClose(handle);
      Print("Data successfully written to: ", fileName);
   }
}

//+------------------------------------------------------------------+
//| Populate training data (with Min-Max normalization)              |
//+------------------------------------------------------------------+
void InitBars()
{
   minPrice = 999999.0;
   maxPrice = -999999.0;
   
   for(int i=0; i<trainSize; i++)
   {
      int id = trainSize - i;
      double price = iMA(NULL, 0, 1, id, MODE_SMA, PRICE_CLOSE, 0);
      if(price < minPrice) minPrice = price;
      if(price > maxPrice) maxPrice = price;
   }
   
   if(maxPrice == minPrice)
   {
      maxPrice = minPrice + 0.001; // Avoid division by zero
   }

   for(int i=0; i<trainSize; i++)
   {
      int id = trainSize - i;
      double price = iMA(NULL, 0, 1, id, MODE_SMA, PRICE_CLOSE, 0);
      etalons[i] = (price - minPrice) / (maxPrice - minPrice);
   }
}

//+------------------------------------------------------------------+
//| Neural network training function (Backpropagation)               |
//+------------------------------------------------------------------+
void Train()
{
   if(countTeaches > maxTeach)
   {
      // return;
   }

   fastPassCounter++;
   if(allowSkipTrain == true && errGlobal < tradeErrThresold && fastPassCounter < 88)
   {
      return;
   }
   
   fastPassCounter = 0;  

   for(int pox=0; pox < pass; pox++)
   {
      countTeaches++;
      errGlobal = 0.0;

      for(int iSample=0; iSample < trainSize - p; iSample++)
      {
         for(int i=0; i < countHiddenNeuron; i++)
         {
            double wSum = 0;
            for(int j=0; j < p; j++)
            {
               double x = etalons[j+iSample];
               wSum += weightsHidden[i][j] * x;
            }
            wSum -= thresoldsHidden[i];
            weightedSums[i] = wSum;
            outputsHidden[i] = Sigmoid(wSum);
         }

         double weightedSumOutputLayer = 0;
         for(int i=0; i < countHiddenNeuron; i++)
         {
            double x = outputsHidden[i];
            weightedSumOutputLayer += weightsOutputLayer[i] * x;
         }
         weightedSumOutputLayer -= thresoldOutputLayer;
         double outputMain = Sigmoid(weightedSumOutputLayer);

         yValues[iSample] = outputMain;

         // Weights & bias update rules
         double a = aStep;
         double errOutput = outputMain - etalons[iSample + p];
         errGlobal += errOutput * errOutput;
         
         double errHiden[countHiddenNeuron];
         for(int i=0; i < countHiddenNeuron; i++)
         {
            errHiden[i] = errOutput * SigmoidDerivative(weightedSumOutputLayer) * weightsOutputLayer[i];
         }

         // Corrected output layer weights update (assigning the calculated value)
         for(int i=0; i < countHiddenNeuron; i++)
         {
            double w = weightsOutputLayer[i] - a * errOutput * SigmoidDerivative(weightedSumOutputLayer) * outputsHidden[i];
            weightsOutputLayer[i] = w;
         }

         thresoldOutputLayer = thresoldOutputLayer + a * errOutput * SigmoidDerivative(weightedSumOutputLayer);

         for(int i=0; i < countHiddenNeuron; i++)
         {
            for(int j=0; j < p; j++)
            {
               double w = weightsHidden[i][j];
               double x = etalons[j + iSample];
               w = w - a * errHiden[i] * SigmoidDerivative(weightedSums[i]) * x;
               weightsHidden[i][j] = w;
            }
            thresoldsHidden[i] = thresoldsHidden[i] + a * errHiden[i] * SigmoidDerivative(weightedSums[i]);
         }
      }

      errGlobal *= 100.0;
   }

   Print("Teach Iteration: ", countTeaches, " === Global Error: ", errGlobal, " Save Timer: ", saveTimer);

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
         if(ovverideSaveFile)
         {
            SaveWeightsData(false);
         }
         else
         {
            SaveWeightsData(true);
         }  
      }  
   }  
}

//+------------------------------------------------------------------+
//| Generate predictions (Autoregressive MLP forecasting)            |
//+------------------------------------------------------------------+
void Predict()
{
   for(int pre=0; pre<predictBars; pre++)
   {
      double outH[countHiddenNeuron];
      for(int i=0; i < countHiddenNeuron; i++)
      {
         double wSum = 0;
         for(int j=0; j < p; j++)
         {
            double x = 0;
            int id = trainSize - p + j + pre;
            if(id >= trainSize)
            {
               x = predictValues[pre-1];
            }
            else
            {
               x = etalons[id];
            }
            wSum += weightsHidden[i][j] * x;
         }
         wSum -= thresoldsHidden[i];
         outH[i] = Sigmoid(wSum);
      }

      double weightedSumOutputLayer = 0;
      for(int i=0; i < countHiddenNeuron; i++)
      {
         double x = outH[i];
         weightedSumOutputLayer += weightsOutputLayer[i] * x;
      }
      weightedSumOutputLayer -= thresoldOutputLayer;
      double outputMain = Sigmoid(weightedSumOutputLayer);

      predictValues[pre] = outputMain;
   }

   if(errGlobal < tradeErrThresold)
   {
      if(isRealtime)
      {
         if(trainRequired)
         {
            Trade();
         }
      }
      else
      {
         Trade();
      }
   }
}

//+------------------------------------------------------------------+
//| Execute trades based on predictions                              |
//+------------------------------------------------------------------+
void Trade()
{
   if(OrdersTotal() > 0)
   {
      // return;
   }

   double rangeP = maxPrice - minPrice;
   if(rangeP <= 0) return;

   // Denormalize current price and predictions
   double curPrice = yValues[trainSize - p - 1] * rangeP + minPrice;
   if(predictShift > 0)
   {
      curPrice = predictValues[predictShift - 1] * rangeP + minPrice;
   }

   double sumLow = 0;
   double sumHigh = 0;
   int countLow = 0;
   int countHigh = 0;

   int total = predictBars > countPredictAnalysys ? countPredictAnalysys : predictBars;

   for(int i=0; i<total; i++)
   {
      double predVal = predictValues[i + predictShift] * rangeP + minPrice;
      if(predVal > curPrice)
      {
         countHigh++;
         sumHigh += predVal;
      }
      else
      {
         countLow++;
         sumLow += predVal;
      }
   }

   int pipsHigh = 0;
   int pipsLow = 0;

   if(countHigh > 0)
   {
      double averageHigh = sumHigh / (double)countHigh;
      pipsHigh = MathRound((averageHigh - curPrice) / _Point);
   }
   if(countLow > 0)
   {
      double averageLow = sumLow / (double)countLow;
      pipsLow = MathRound((curPrice - averageLow) / _Point);
   }

   trainRequired = false;
   if(pipsHigh - pipsLow > orderThresold)
   {
      int t = pipsHigh > takeProfit ? takeProfit : pipsHigh * tpMultiplier;
      int s = pipsLow < stopLoss ? stopLoss : pipsLow;
      double sl = Bid - s * _Point;
      double tp = Bid + t * _Point;

      ticket = OrderSend(_Symbol, OP_BUY, 0.1, Ask, 5, sl, tp, "");
      trainRequired = false;
   }
   else if(pipsLow - pipsHigh > orderThresold)
   {
      int t = pipsLow > takeProfit ? takeProfit : pipsLow * tpMultiplier;
      int s = pipsHigh < stopLoss ? stopLoss : pipsHigh;
      double sl = Ask + s * _Point;
      double tp = Ask - t * _Point;

      ticket = OrderSend(_Symbol, OP_SELL, 0.1, Bid, 5, sl, tp, "");
      trainRequired = false;
   }
}

//+------------------------------------------------------------------+
//| OnTick execution handler                                         |
//+------------------------------------------------------------------+
void OnTick()
{
   if(IsNewBar())
   {
      countTryed = 0;
      nonTeacheble = false;
      trainRequired = true;
      InitBars();
   }

   if(trainRequired && !nonTeacheble)
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
}

//+------------------------------------------------------------------+
//| Plot historical actual prices                                    |
//+------------------------------------------------------------------+
void DrawLines()
{
   ObjectsDeleteAll();
   double rangeP = maxPrice - minPrice;

   for(int i=0; i < trainSize - 1; i++)
   {
      double x1 = Time[trainSize - i + predictBars];
      double y1 = (etalons[i] * rangeP + minPrice) + offsetY;
      double x2 = Time[trainSize - i + predictBars - 1];
      double y2 = (etalons[i+1] * rangeP + minPrice) + offsetY;

      int id = ObjectCreate("PriceLine" + i, OBJ_TREND, 0, x1, y1, x2, y2);
      if(id > 0)
      {
         ObjectSetString(0, "PriceLine" + i, OBJPROP_TEXT, "PriceLine" + i);
         ObjectSet("PriceLine" + i, OBJPROP_COLOR, clrBlue);
         ObjectSet("PriceLine" + i, OBJPROP_WIDTH, 5);
         ObjectSet("PriceLine" + i, OBJPROP_RAY, 0);
      }
   }
}

//+------------------------------------------------------------------+
//| Plot historical neural network output predictions                |
//+------------------------------------------------------------------+
void DrawLinesNN()
{
   double rangeP = maxPrice - minPrice;
   int passValue = trainSize - p - 1;

   for(int i=0; i < passValue; i++)
   {
      double x1 = Time[trainSize - i - p + predictBars];
      double y1 = (yValues[i] * rangeP + minPrice) + offsetY;
      double x2 = Time[trainSize - i - p + predictBars - 1];
      double y2 = (yValues[i+1] * rangeP + minPrice) + offsetY;

      int id = ObjectCreate("NN" + i, OBJ_TREND, 0, x1, y1, x2, y2);
      if(id > 0)
      {
         ObjectSetString(0, "NN" + i, OBJPROP_TEXT, "NN" + i);
         ObjectSet("NN" + i, OBJPROP_COLOR, clrYellow);
         ObjectSet("NN" + i, OBJPROP_WIDTH, 3);
         ObjectSet("NN" + i, OBJPROP_RAY, 0);
      }
   }
}

//+------------------------------------------------------------------+
//| Plot forecasted future prices                                    |
//+------------------------------------------------------------------+
void DrawLinesPredict()
{
   double rangeP = maxPrice - minPrice;
   for(int i=0; i < predictBars; i++)
   {
      double x1 = Time[predictBars - i + 1];
      double y1 = i - 1 < 0 ? (yValues[trainSize - p - 1] * rangeP + minPrice) : (predictValues[i-1] * rangeP + minPrice);
      double x2 = Time[predictBars - i];
      double y2 = predictValues[i] * rangeP + minPrice;

      int id = ObjectCreate("Predict" + i, OBJ_TREND, 0, x1, y1, x2, y2);
      if(id > 0)
      {
         ObjectSetString(0, "Predict" + i, OBJPROP_TEXT, "Predict" + i);
         ObjectSet("Predict" + i, OBJPROP_COLOR, clrLimeGreen);
         ObjectSet("Predict" + i, OBJPROP_WIDTH, 3);
         ObjectSet("Predict" + i, OBJPROP_RAY, 0);
      }
   }
}

//+------------------------------------------------------------------+
//| Clear custom chart lines                                         |
//+------------------------------------------------------------------+
void ClearChart()
{
   for(int i=0; i < trainSize - p; i++)
   {
      int id = ObjectFind("PriceLine" + i);
      if(id != -1)
      {
         ObjectDelete("PriceLine" + i);
      }
   }
}

//+------------------------------------------------------------------+
//| Generate random double value                                     |
//+------------------------------------------------------------------+
double RandomDouble(double minVal, double maxVal)
{
   return minVal + (maxVal - minVal) * MathRand() / 32767.0;
}

//+------------------------------------------------------------------+
//| Leaky ReLU Activation Function                                   |
//+------------------------------------------------------------------+
double Sigmoid(double x)
{
   if(x > 0)
      return x;
   else
      return 0.0001f;
}

//+------------------------------------------------------------------+
//| Leaky ReLU Derivative Function                                   |
//+------------------------------------------------------------------+
double SigmoidDerivative(double y)
{
   if(y > 0)
      return 1.0;
   else
      return 0.0001f;
}

//+------------------------------------------------------------------+
//| Check if new bar opened                                          |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   static int nBars = 0;
   if(nBars != Bars)
   {
      nBars = Bars;
      return true;
   }
   return false;
}
