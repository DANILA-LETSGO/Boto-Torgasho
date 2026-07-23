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
#define trainSize 3600
#define MAX_OUTPUTS 32 // потолок числа выходов (размер статических массивов)

#define CANDLES_N 9
#define CORR_N 3

input double tpMultiplier = 1.0;
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
input int pass = 1;
input int maxEpochsPerBar = 150; // Максимальное число эпох на один бар (защита от вечного обучения)
input bool forceTradeOnEpochLimit = false; // Разрешить торговлю при достижении лимита эпох, даже если ошибка высока
int epochsTrainedThisBar = 0;
// v7.9: число выходных нейронов = горизонт прогноза в барах. Выход o предсказывает
// СУММАРНЫЙ ход цены за o+1 будущих баров (кумулятивно), а не дельту отдельного бара.
// Должно быть >= predictShift + countPredictAnalysys (иначе Trade() не хватит горизонта)
// и <= MAX_OUTPUTS. Авторегрессии больше нет: один проход сети — весь прогноз.
input int countOutputBars = 12;
input int countPredictAnalysys = 6;
input int predictShift = 3;
input double wRange = 0.35;// диапозон случайных значений весов 0.9;

input bool allowSkipTrain = false;
input int skipTrainMaxPasses = 24;
input double aStep = 0.01;
input double momentum = 0.9; // Инерция для SGD
input double gradClipLimit = 3.0; // Ограничение |errOutput| перед градиентом (ед. локального диапазона окна) — защита от взрыва весов
input bool useWeeklyReset = false; // Сбрасывать веса раз в неделю (иначе — раз в день)
input int validationSize = 0; // Сколько последних примеров спрятать от обучения для честной проверки (0 = выкл)
input bool useEarlyStopping = false; // Останавливать обучение по экзамену и брать лучшие веса (нужен validationSize > 0)
input int earlyStopPatience = 10; // Сколько эпох терпеть без улучшения экзамена, прежде чем остановиться
input bool tradeOnlyIfBeatsNaive = false; // Торговать только если сеть обошла тупой прогноз
input double beatNaiveRatio = 1.0; // Насколько надо обойти тупой прогноз (1.0 = просто лучше, 0.98 = лучше на 2%)
input double skipTrainRatio = 1.0; // Пропускать обучение, пока экзамен лучше тупого прогноза во столько раз (нужен allowSkipTrain)
input bool shuffleTraining = false; // Перемешивать порядок учебных примеров каждую эпоху
input int examDirBars = 6; // Направление в экзамене: горизонт в барах (как торговый countPredictAnalysys; 0 = выкл)
input bool onlyOneOrder = true;
input bool closeOnReverseSignal = true;
input int magicNumber = 123456;
input bool useRandomSeed = false;  // использовать фиксированный seed для генератора
input int randomSeed = 42;         // значение seed (активно только если useRandomSeed = true)
input bool useDaySeed = false;     // прибавлять к сиду номер дня: каждый день новые, но воспроизводимые стартовые веса

input int checkLastNTrades = 5;          // количество последних сделок для анализа
input int maxLossThresholdPips = 1000;   // порог суммарного убытка (в пунктах) для сброса весов

datetime lastResetTime = 0;

double weightsHidden[countHiddenNeuron][p+10+CANDLES_N*3+CORR_N*2];
double thresoldsHidden[countHiddenNeuron];
double weightedSums[countHiddenNeuron];
double outputsHidden[countHiddenNeuron];
double weightsOutputLayer[MAX_OUTPUTS][countHiddenNeuron];
double thresoldOutputLayer[MAX_OUTPUTS];

double vWeightsHidden[countHiddenNeuron][p+10+CANDLES_N*3+CORR_N*2];
double vThresoldsHidden[countHiddenNeuron];
double vWeightsOutputLayer[MAX_OUTPUTS][countHiddenNeuron];
double vThresoldOutputLayer[MAX_OUTPUTS];

double yValues[trainSize - p];
double etalons[trainSize];
double rawPrices[trainSize];


double offsetY = 0;// 0.195;
int countTeaches = 0;
double errIncrease = 0;

int ticket;
bool showGraphics = true; // false в оптимизации / тесте без визуализации

//---- Честная проверка: последние validSamples примеров скрыты от обучения ----
int trainSamples = trainSize - p; // примеры, на которых сеть учится
int validSamples = 0;             // примеры, которых сеть не видит (экзамен)

double errValidation = 0; // ошибка сети на спрятанных примерах
double errNaive = 0;      // ошибка тупого прогноза "цена не изменится" на них же
double errNaiveTrain = 0; // ошибка тупого прогноза на учебных примерах (для честного сравнения с домашкой)

// Направление: на скольких спрятанных барах сеть угадала знак движения цены
int dirCorrectBar = 0; // угадано на текущем баре
int dirTotalBar = 0;   // всего спрятанных примеров с ненулевым движением на текущем баре

// То же, но на торговом горизонте: суммарный ход за examDirBars баров вперёд
int dirHorCorrectBar = 0;
int dirHorTotalBar = 0;

// Сигнал как в Trade(): пропуск predictShift баров, опора, перевес выше/ниже
int dirSigCorrectBar = 0;
int dirSigTotalBar = 0;

// Громкий сигнал: перевес прошёл торговый порог orderThresold — только такие ведут к сделке
int dirLoudCorrectBar = 0;
int dirLoudTotalBar = 0;

// Накопители для итогового отчёта в конце теста
double sumErrTrain = 0;
double sumErrValidation = 0;
double sumErrNaive = 0;
double sumErrNaiveTrain = 0;
int sumDirCorrect = 0;
int sumDirTotal = 0;
int sumDirHorCorrect = 0;
int sumDirHorTotal = 0;
int sumDirSigCorrect = 0;
int sumDirSigTotal = 0;
int sumDirLoudCorrect = 0;
int sumDirLoudTotal = 0;
int validationBars = 0;

//---- Порядок подачи учебных примеров внутри эпохи (для перемешивания) ----
int trainOrder[];

//---- Лучшие веса за текущий бар: те, на которых экзамен был сдан лучше всего ----
double bestWeightsHidden[countHiddenNeuron][p+10+CANDLES_N*3+CORR_N*2];
double bestThresoldsHidden[countHiddenNeuron];
double bestWeightsOutputLayer[MAX_OUTPUTS][countHiddenNeuron];
double bestThresoldOutputLayer[MAX_OUTPUTS];

double bestErrValidation = 999999; // лучшая (наименьшая) ошибка на экзамене
bool hasBestWeights = false;
int epochsSinceImprove = 0;        // сколько эпох подряд экзамен не улучшался
bool skipTrainingThisBar = false;  // сеть на этом баре и так справляется, обучение не нужно

bool ebobo;
double errGlobal = 999;

input double tradeErrThresold = 8.0;// Порог ошибки для торговли 0.13
input bool useDynamicError = false; // Динамический порог ошибки (ATR 14 / ATR 100)
double currentTradeErrThreshold = 0.235;
double teachErrThresold = 3.5;
int fastPassCounter;

double predictValues[MAX_OUTPUTS]; // по-барные дельты, разложенные из кумулятивных выходов

int examDirBarsEff = 0; // examDirBars, прижатый к countOutputBars (input менять нельзя)

//---- Кэш признаков сэмплов (не зависят от прохода обучения pox, только от бара) ----
bool featuresCacheValid = false;
double cacheLocalMin[trainSize - p];
double cacheRange[trainSize - p];
double cacheRsiInput[trainSize - p];
double cacheRsiInput2[trainSize - p];
double cacheAtrInput[trainSize - p];
double cacheAtrInput2[trainSize - p];
double cacheMaInput[trainSize - p];
double cacheMaInput2[trainSize - p];
double cacheCandleInputs[trainSize - p][CANDLES_N*3];
double cacheCorr1Inputs[trainSize - p][CORR_N];
double cacheCorr2Inputs[trainSize - p][CORR_N];
double cacheSinTime[trainSize - p];
double cacheCosTime[trainSize - p];
double cacheSinDay[trainSize - p];
double cacheCosDay[trainSize - p];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // В оптимизации (и в тесте без визуализации) график никто не видит, а отрисовка
   // тысяч объектов на каждый бар — основная статья расходов по времени прохода.
   showGraphics = !IsOptimization() && (!IsTesting() || IsVisualMode());

   if(countOutputBars < 1 || countOutputBars > MAX_OUTPUTS)
     {
      Alert("countOutputBars должен быть от 1 до ", MAX_OUTPUTS, ", а не ", countOutputBars);
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(countOutputBars < predictShift + countPredictAnalysys)
     {
      Alert("countOutputBars (", countOutputBars, ") должен быть >= predictShift + countPredictAnalysys (",
            predictShift + countPredictAnalysys, "), иначе Trade() не хватит горизонта прогноза");
      return(INIT_PARAMETERS_INCORRECT);
     }
   examDirBarsEff = examDirBars;
   if(examDirBarsEff > countOutputBars) examDirBarsEff = countOutputBars;

   // Делим примеры на "учебные" и "экзаменационные". Экзаменационные — самые свежие,
   // они ближе всего к тому, что бот увидит в реальной торговле. Сеть их не видит.
   // Последним countOutputBars-1 примерам не хватает будущего для всех кумулятивных
   // целей — они не участвуют ни в обучении, ни в экзамене.
   int usableSamples = (trainSize - p) - (countOutputBars - 1);
   int vSize = validationSize;
   if(vSize < 0) vSize = 0;
   if(vSize > usableSamples / 2) vSize = usableSamples / 2; // не больше половины
   validSamples = vSize;
   trainSamples = usableSamples - validSamples;

   sumErrTrain = 0;
   sumErrValidation = 0;
   sumErrNaive = 0;
   sumErrNaiveTrain = 0;
   sumDirCorrect = 0;
   sumDirTotal = 0;
   sumDirHorCorrect = 0;
   sumDirHorTotal = 0;
   sumDirSigCorrect = 0;
   sumDirSigTotal = 0;
   sumDirLoudCorrect = 0;
   sumDirLoudTotal = 0;
   validationBars = 0;

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

   if(useRandomSeed)
     {
      // useDaySeed: у каждого дня (недели) свои стартовые веса, но воспроизводимые —
      // тест, запущенный с любой даты, даст для конкретного календарного дня те же веса.
      if(useDaySeed) MathSrand(randomSeed + SeedPeriodKey());
      else           MathSrand(randomSeed);
     }
   else MathSrand(GetTickCount());

   // 1. Generate base weights exactly as in v7.8
   for(int i=0; i<countHiddenNeuron; i++)
     {
      for(int j=0; j<=p+1; j++)
        {
         weightsHidden[i][j] = RandomDouble(-wRange, wRange);
        }
      thresoldsHidden[i] = RandomDouble(-wRange, wRange);
     }
   // Выходной слой: своя строка весов и порог на каждый выход-горизонт
   for(int o=0; o<countOutputBars; o++)
     {
      for(int i=0; i<countHiddenNeuron; i++)
        {
         weightsOutputLayer[o][i] = RandomDouble(-wRange, wRange);
        }
      thresoldOutputLayer[o] = RandomDouble(-wRange, wRange);
     }

   // 2. Generate extra weights from the tail of generator
   for(int i=0; i<countHiddenNeuron; i++)
     {
      for(int j=p+2; j<=p+9+CANDLES_N*3+CORR_N*2; j++)
        {
         weightsHidden[i][j] = RandomDouble(-wRange, wRange);
        }
     }
     
   for(int i=0; i<countHiddenNeuron; i++)
     {
      for(int j=0; j<=p+9+CANDLES_N*3+CORR_N*2; j++)
        {
         vWeightsHidden[i][j] = 0.0;
        }
      vThresoldsHidden[i] = 0.0;
     }
   for(int o=0; o<MAX_OUTPUTS; o++)
     {
      for(int i=0; i<countHiddenNeuron; i++)
        {
         vWeightsOutputLayer[o][i] = 0.0;
        }
      vThresoldOutputLayer[o] = 0.0;
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
   featuresCacheValid = false;
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

// Признаки сэмплов зависят только от номера сэмпла (etalons/индикаторы), а не от
// текущих весов сети или номера прохода обучения — считаем их один раз на бар
// и переиспользуем во всех проходах Train(), вместо пересчёта на каждый pox.
//
// atrInput/atrInput2/candleInputs пропускаются через MathArctan (как maInput), а не
// берутся как "сырое" отношение к range — иначе на узком range (спокойный рынок) и
// резком ATR/свече (спайк/гэп после новости) значение входа улетает на порядки,
// вызывая взрыв активации в LeakyReLU и далее взрыв градиента/весов (NaN).
void PrecomputeSampleFeatures()
  {
   if(featuresCacheValid) return;

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
      cacheLocalMin[iSample] = localMin;
      cacheRange[iSample] = range;

      double rsi = CalcRSI(etalons, iSample + p - 1, 14);
      cacheRsiInput[iSample] = (rsi - 50.0) / 100.0;
      double rsi2 = CalcRSI(etalons, iSample + p - 1, 28);
      cacheRsiInput2[iSample] = (rsi2 - 50.0) / 100.0;

      int shift = trainSize - (iSample + p - 1);
      double atrInput = 0;
      double atrInput2 = 0;
      if(useAtrAsInput)
        {
         double atr = iATR(NULL, 0, 14, shift);
         atrInput = 2.0 * MathArctan(atr / range) / 3.1415926535;
         double atr2 = iATR(NULL, 0, 100, shift);
         atrInput2 = 2.0 * MathArctan(atr2 / range) / 3.1415926535;
        }
      cacheAtrInput[iSample] = atrInput;
      cacheAtrInput2[iSample] = atrInput2;

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
      cacheMaInput[iSample] = maInput;
      cacheMaInput2[iSample] = maInput2;

      for(int c=0; c<CANDLES_N*3; c++) cacheCandleInputs[iSample][c] = 0;
      if(useCandlesAsInput)
        {
         for(int c=0; c<CANDLES_N; c++)
           {
            int cShift = shift + c;
            double cOpen = iOpen(NULL, 0, cShift);
            double cHigh = iHigh(NULL, 0, cShift);
            double cLow = iLow(NULL, 0, cShift);
            double cClose = iClose(NULL, 0, cShift);

            cacheCandleInputs[iSample][c*3] = 2.0 * MathArctan((cClose - cOpen) / range) / 3.1415926535;
            cacheCandleInputs[iSample][c*3+1] = 2.0 * MathArctan((cHigh - MathMax(cOpen, cClose)) / range) / 3.1415926535;
            cacheCandleInputs[iSample][c*3+2] = 2.0 * MathArctan((MathMin(cOpen, cClose) - cLow) / range) / 3.1415926535;
           }
        }

      for(int c=0; c<CORR_N; c++) { cacheCorr1Inputs[iSample][c] = 0; cacheCorr2Inputs[iSample][c] = 0; }
      if(useCorrelationAsInput)
        {
         for(int c=0; c<CORR_N; c++)
           {
            int cShift = shift + c;
            double rsi1 = iRSI(correlationSymbol1, 0, 14, PRICE_CLOSE, cShift);
            double rsiCorr2 = iRSI(correlationSymbol2, 0, 14, PRICE_CLOSE, cShift);
            cacheCorr1Inputs[iSample][c] = (rsi1 - 50.0) / 100.0;
            cacheCorr2Inputs[iSample][c] = (rsiCorr2 - 50.0) / 100.0;
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
      cacheSinTime[iSample] = sinTime;
      cacheCosTime[iSample] = cosTime;
      cacheSinDay[iSample] = sinDay;
      cacheCosDay[iSample] = cosDay;
     }

   featuresCacheValid = true;
  }

void CalculateError()
  {
   PrecomputeSampleFeatures();

   errGlobal = 0;
   for(int iSample=0; iSample<trainSamples; iSample++)
     {
      double localMin = cacheLocalMin[iSample];
      double range = cacheRange[iSample];

      for(int i=0; i<countHiddenNeuron; i++)
        {
         double wSum = 0;
         for(int j=0; j<p; j++)
           {
            double x = (etalons[j+iSample] - localMin) / range;
            wSum += weightsHidden[i][j] * x;
           }

         wSum += weightsHidden[i][p] * cacheRsiInput[iSample];
         wSum += weightsHidden[i][p+1] * cacheRsiInput2[iSample];
         wSum += weightsHidden[i][p+2] * cacheAtrInput[iSample];
         wSum += weightsHidden[i][p+3] * cacheAtrInput2[iSample];
         wSum += weightsHidden[i][p+4] * cacheMaInput[iSample];
         wSum += weightsHidden[i][p+5] * cacheMaInput2[iSample];
         for(int c=0; c<CANDLES_N*3; c++)
           {
            wSum += weightsHidden[i][p+6+c] * cacheCandleInputs[iSample][c];
           }
         wSum += weightsHidden[i][p+6+CANDLES_N*3] * cacheSinTime[iSample];
         wSum += weightsHidden[i][p+7+CANDLES_N*3] * cacheCosTime[iSample];
         wSum += weightsHidden[i][p+8+CANDLES_N*3] * cacheSinDay[iSample];
         wSum += weightsHidden[i][p+9+CANDLES_N*3] * cacheCosDay[iSample];
         for(int c=0; c<CORR_N; c++)
           {
            wSum += weightsHidden[i][p+10+CANDLES_N*3+c] * cacheCorr1Inputs[iSample][c];
            wSum += weightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] * cacheCorr2Inputs[iSample][c];
           }

         wSum -= thresoldsHidden[i];
         weightedSums[i] = wSum;
         outputsHidden[i] = LeakyReLU(wSum);
        }

      // Выход o предсказывает суммарный ход за o+1 баров (кумулятивная цель)
      double sumSqErr = 0;
      double targetCum = 0;
      for(int o=0; o<countOutputBars; o++)
        {
         double outO = 0;
         for(int i=0; i<countHiddenNeuron; i++)
           {
            outO += weightsOutputLayer[o][i] * outputsHidden[i];
           }
         outO -= thresoldOutputLayer[o]; // Linear Activation

         // Исправление Бага 3: обновляем массив предсказаний при пропуске обучения
         if(o == 0) yValues[iSample] = outO * range + localMin;

         // Целевое значение тоже нормализуется относительно текущего окна
         targetCum += etalons[iSample + p + o];
         double targetNorm = (targetCum - localMin) / range;
         double errOutput = outO - targetNorm;
         sumSqErr += errOutput * errOutput;
        }

      // Чистая нормализованная ошибка, усреднённая по выходам —
      // масштаб errGlobal не зависит от countOutputBars
      errGlobal += sumSqErr / countOutputBars;
     }
   // Переводим в удобный масштаб (например, умножаем на 100, чтобы значения были около 0.1 - 1.0)
   errGlobal = (errGlobal / trainSamples) * 100;
  }

//+------------------------------------------------------------------+
//| Честный экзамен: ошибка на примерах, которых сеть не видела       |
//|                                                                   |
//| Считаем два числа на одних и тех же спрятанных примерах:          |
//|  errValidation — как ошибается сеть;                              |
//|  errNaive      — как ошибается тупейший прогноз "цена не изменится"|
//|                  (то есть предсказание "изменение = 0").          |
//| Если errValidation не лучше errNaive — сеть не предсказывает      |
//| ничего, она лишь зазубрила обучающие примеры.                     |
//| Веса здесь не трогаем — это только замер.                         |
//+------------------------------------------------------------------+
void CalculateValidationError()
  {
   errValidation = 0;
   errNaive = 0;
   errNaiveTrain = 0;
   dirCorrectBar = 0;
   dirTotalBar = 0;
   dirHorCorrectBar = 0;
   dirHorTotalBar = 0;
   dirSigCorrectBar = 0;
   dirSigTotalBar = 0;
   dirLoudCorrectBar = 0;
   dirLoudTotalBar = 0;
   if(validSamples <= 0) return;

   PrecomputeSampleFeatures();

   for(int iSample = trainSamples; iSample < trainSamples + validSamples; iSample++)
     {
      double localMin = cacheLocalMin[iSample];
      double range = cacheRange[iSample];

      double outH[countHiddenNeuron];
      for(int i=0; i<countHiddenNeuron; i++)
        {
         double wSum = 0;
         for(int j=0; j<p; j++)
           {
            double x = (etalons[j+iSample] - localMin) / range;
            wSum += weightsHidden[i][j] * x;
           }

         wSum += weightsHidden[i][p] * cacheRsiInput[iSample];
         wSum += weightsHidden[i][p+1] * cacheRsiInput2[iSample];
         wSum += weightsHidden[i][p+2] * cacheAtrInput[iSample];
         wSum += weightsHidden[i][p+3] * cacheAtrInput2[iSample];
         wSum += weightsHidden[i][p+4] * cacheMaInput[iSample];
         wSum += weightsHidden[i][p+5] * cacheMaInput2[iSample];
         for(int c=0; c<CANDLES_N*3; c++)
           {
            wSum += weightsHidden[i][p+6+c] * cacheCandleInputs[iSample][c];
           }
         wSum += weightsHidden[i][p+6+CANDLES_N*3] * cacheSinTime[iSample];
         wSum += weightsHidden[i][p+7+CANDLES_N*3] * cacheCosTime[iSample];
         wSum += weightsHidden[i][p+8+CANDLES_N*3] * cacheSinDay[iSample];
         wSum += weightsHidden[i][p+9+CANDLES_N*3] * cacheCosDay[iSample];
         for(int c=0; c<CORR_N; c++)
           {
            wSum += weightsHidden[i][p+10+CANDLES_N*3+c] * cacheCorr1Inputs[iSample][c];
            wSum += weightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] * cacheCorr2Inputs[iSample][c];
           }

         wSum -= thresoldsHidden[i];
         outH[i] = LeakyReLU(wSum);
        }

      // Все выходы сразу: outCum[o] — прогноз суммарного хода за o+1 баров (в ценах),
      // actCum[o] — реальный суммарный ход за те же бары.
      double outCum[MAX_OUTPUTS];
      double actCum[MAX_OUTPUTS];
      double sumSqNet = 0;
      double sumSqNv = 0;
      // Тупой прогноз: изменения цены не будет — суммарный ход 0 на любом горизонте.
      // Нормализуем его тем же способом, что и всё остальное, чтобы числа были сравнимы.
      double naiveNorm = (0.0 - localMin) / range;
      double targetCum = 0;
      for(int o=0; o<countOutputBars; o++)
        {
         double outO = 0;
         for(int i=0; i<countHiddenNeuron; i++)
           {
            outO += weightsOutputLayer[o][i] * outH[i];
           }
         outO -= thresoldOutputLayer[o];

         targetCum += etalons[iSample + p + o];
         double targetNorm = (targetCum - localMin) / range;

         double errNet = outO - targetNorm;
         sumSqNet += errNet * errNet;
         double errNv = naiveNorm - targetNorm;
         sumSqNv += errNv * errNv;

         outCum[o] = outO * range + localMin;
         actCum[o] = targetCum;
        }
      errValidation += sumSqNet / countOutputBars;
      errNaive += sumSqNv / countOutputBars;

      // Направление: угадала ли сеть знак движения цены (в пунктах, без нормализации).
      // Тупой прогноз направления не даёт вообще, монетка даёт ~50%.
      double predDelta = outCum[0];              // прогноз изменения цены за 1 бар
      double actualDelta = etalons[iSample + p]; // реальное изменение цены
      if(actualDelta != 0)
        {
         dirTotalBar++;
         if((predDelta > 0 && actualDelta > 0) || (predDelta < 0 && actualDelta < 0))
            dirCorrectBar++;
        }

      // Горизонтные метрики: роллаута больше нет — выходы сети напрямую дают
      // суммарный ход на каждом горизонте. Два вердикта:
      //  напр.Nбар — знак суммарного хода цены за examDirBars баров;
      //  сигнал    — точная копия расчёта Trade(): пропустить predictShift баров,
      //              поставить опору, средняя высота точек пути выше опоры минус
      //              средняя глубина точек ниже, знак перевеса.
      if(examDirBarsEff >= 1)
        {
         // --- Вердикт 1: суммарный ход за examDirBars баров ---
         double predSum = outCum[examDirBarsEff - 1];
         double actualSum = actCum[examDirBarsEff - 1];
         if(actualSum != 0)
           {
            dirHorTotalBar++;
            if((predSum > 0 && actualSum > 0) || (predSum < 0 && actualSum < 0))
               dirHorCorrectBar++;
           }

         // --- Вердикт 2: сигнал как в Trade() ---
         int Hsig = predictShift + countPredictAnalysys;
         if(Hsig > countOutputBars) Hsig = countOutputBars;

         // Путь цены относительно опоры (опора = прогноз после predictShift баров эха)
         double baseCum = (predictShift > 0) ? outCum[predictShift - 1] : 0;
         double sumHighRel = 0;
         double sumLowRel = 0;
         int cntHigh = 0;
         int cntLow = 0;
         for(int i2=predictShift; i2<Hsig; i2++)
           {
            double pathRel = outCum[i2] - baseCum;
            if(pathRel > 0) { cntHigh++; sumHighRel += pathRel; }
            else            { cntLow++;  sumLowRel += pathRel; }
           }
         double avgHigh = (cntHigh > 0) ? sumHighRel / cntHigh : 0;   // средняя высота над опорой
         double avgLow  = (cntLow > 0) ? (-sumLowRel / cntLow) : 0;   // средняя глубина под опорой
         double signalVal = avgHigh - avgLow; // >0 — сигнал вверх, <0 — вниз (как pipsHigh - pipsLow)

         // Реальный ход цены с текущего момента до конца сигнального окна:
         // сделка открывается сейчас, поэтому бары эха тоже считаются.
         double actualSig = (Hsig > 0) ? actCum[Hsig - 1] : 0;

         if(actualSig != 0 && Hsig > predictShift)
           {
            dirSigTotalBar++;
            if((signalVal > 0 && actualSig > 0) || (signalVal < 0 && actualSig < 0))
               dirSigCorrectBar++;

            // Громкий сигнал: перевес больше торгового порога (в пунктах, как в Trade).
            // Динамический порог здесь не учитываем — берём базовый orderThresold.
            if(MathAbs(signalVal) > orderThresold * _Point)
              {
               dirLoudTotalBar++;
               if((signalVal > 0 && actualSig > 0) || (signalVal < 0 && actualSig < 0))
                  dirLoudCorrectBar++;
              }
           }
        }
     }

   errValidation = (errValidation / validSamples) * 100;
   errNaive = (errNaive / validSamples) * 100;

   // Тупой прогноз на учебных примерах — честная планка для ошибки "обучение".
   // Сеть здесь не гоняем, только считаем цену молчания на домашке.
   for(int iSample = 0; iSample < trainSamples; iSample++)
     {
      double localMin = cacheLocalMin[iSample];
      double range = cacheRange[iSample];
      double naiveNorm = (0.0 - localMin) / range;
      double sumSqNv = 0;
      double targetCum = 0;
      for(int o=0; o<countOutputBars; o++)
        {
         targetCum += etalons[iSample + p + o];
         double targetNorm = (targetCum - localMin) / range;
         double errNv = naiveNorm - targetNorm;
         sumSqNv += errNv * errNv;
        }
      errNaiveTrain += sumSqNv / countOutputBars;
     }
   errNaiveTrain = (errNaiveTrain / trainSamples) * 100;
  }

//+------------------------------------------------------------------+
//| Запомнить веса как лучшие (экзамен на них сдан лучше всего)       |
//+------------------------------------------------------------------+
void SaveBestWeights()
  {
   for(int i=0; i<countHiddenNeuron; i++)
     {
      for(int j=0; j<=p+9+CANDLES_N*3+CORR_N*2; j++)
        {
         bestWeightsHidden[i][j] = weightsHidden[i][j];
        }
      bestThresoldsHidden[i] = thresoldsHidden[i];
     }
   for(int o=0; o<countOutputBars; o++)
     {
      for(int i=0; i<countHiddenNeuron; i++)
        {
         bestWeightsOutputLayer[o][i] = weightsOutputLayer[o][i];
        }
      bestThresoldOutputLayer[o] = thresoldOutputLayer[o];
     }
   hasBestWeights = true;
  }

//+------------------------------------------------------------------+
//| Вернуть лучшие веса. После этого момента сеть уже начала зубрить, |
//| поэтому торгуем и рисуем именно на них, а не на последних.        |
//+------------------------------------------------------------------+
void RestoreBestWeights()
  {
   if(!hasBestWeights) return;

   for(int i=0; i<countHiddenNeuron; i++)
     {
      for(int j=0; j<=p+9+CANDLES_N*3+CORR_N*2; j++)
        {
         weightsHidden[i][j] = bestWeightsHidden[i][j];
        }
      thresoldsHidden[i] = bestThresoldsHidden[i];
     }
   for(int o=0; o<countOutputBars; o++)
     {
      for(int i=0; i<countHiddenNeuron; i++)
        {
         weightsOutputLayer[o][i] = bestWeightsOutputLayer[o][i];
        }
      thresoldOutputLayer[o] = bestThresoldOutputLayer[o];
     }
  }

//+------------------------------------------------------------------+
//| Порядок подачи учебных примеров на одну эпоху                     |
//| shuffleTraining = false -> 0,1,2,... (по порядку, как всегда было) |
//| shuffleTraining = true  -> те же примеры, но в случайном порядке   |
//+------------------------------------------------------------------+
void BuildTrainOrder()
  {
   if(ArraySize(trainOrder) != trainSamples)
      ArrayResize(trainOrder, trainSamples);

   for(int i=0; i<trainSamples; i++)
      trainOrder[i] = i;

   if(!shuffleTraining) return;

   // Тасование Фишера-Йетса. При выключенной галочке MathRand не трогаем вообще,
   // так что старое поведение воспроизводится бит в бит.
   for(int i=trainSamples-1; i>0; i--)
     {
      int j = MathRand() % (i+1);
      int tmp = trainOrder[i];
      trainOrder[i] = trainOrder[j];
      trainOrder[j] = tmp;
     }
  }

//+------------------------------------------------------------------+
//| Train with Rolling Window                                        |
//+------------------------------------------------------------------+
void Train()
  {
   
   // Старый пропуск обучения смотрит на ошибку по домашке и на порог, который
   // меняет масштаб при смене окна. При включённой ранней остановке он не нужен:
   // там свой пропуск, основанный на экзамене (решение принимается в OnTick).
   if(!useEarlyStopping
      && allowSkipTrain == true
      && errGlobal < currentTradeErrThreshold
      && fastPassCounter < skipTrainMaxPasses)
     {
      return;
     }

   fastPassCounter = 0;

   PrecomputeSampleFeatures();

   for(int pox=0; pox < pass; pox++)
     {
      countTeaches++;
      errGlobal = 0;

      BuildTrainOrder();

      // Учимся только на учебных примерах. Последние validSamples сеть не видит —
      // они отложены на экзамен (CalculateValidationError).
      for(int k=0; k<trainSamples; k++)
        {
         int iSample = trainOrder[k];

         double localMin = cacheLocalMin[iSample];
         double range = cacheRange[iSample];

         double rsiInput = cacheRsiInput[iSample];
         double rsiInput2 = cacheRsiInput2[iSample];
         double atrInput = cacheAtrInput[iSample];
         double atrInput2 = cacheAtrInput2[iSample];
         double maInput = cacheMaInput[iSample];
         double maInput2 = cacheMaInput2[iSample];
         double sinTime = cacheSinTime[iSample];
         double cosTime = cacheCosTime[iSample];
         double sinDay = cacheSinDay[iSample];
         double cosDay = cacheCosDay[iSample];

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
               wSum += weightsHidden[i][p+6+c] * cacheCandleInputs[iSample][c];
              }
            wSum += weightsHidden[i][p+6+CANDLES_N*3] * sinTime;
            wSum += weightsHidden[i][p+7+CANDLES_N*3] * cosTime;
            wSum += weightsHidden[i][p+8+CANDLES_N*3] * sinDay;
            wSum += weightsHidden[i][p+9+CANDLES_N*3] * cosDay;
            for(int c=0; c<CORR_N; c++)
              {
               wSum += weightsHidden[i][p+10+CANDLES_N*3+c] * cacheCorr1Inputs[iSample][c];
               wSum += weightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] * cacheCorr2Inputs[iSample][c];
              }
            
            wSum -= thresoldsHidden[i];
            weightedSums[i] = wSum;
            outputsHidden[i] = LeakyReLU(wSum);
           }

         double a = aStep;

         // Умножаем на range для корректного градиента цены,
         // и делим на 100 * _Point для сохранения масштаба скорости обучения (aStep).
         // Дополнительно делим на число выходов: лосс — среднее по выходам, иначе
         // градиент на скрытый слой рос бы пропорционально countOutputBars
         // (при истории со взрывами весов это прямой путь в NaN).
         double gradScale = range / (100 * _Point) / countOutputBars;

         double sumSqErr = 0;
         double targetCum = 0;
         double errOutClipped[MAX_OUTPUTS];
         for(int o=0; o<countOutputBars; o++)
           {
            double outO = 0;
            for(int i=0; i<countHiddenNeuron; i++)
              {
               outO += weightsOutputLayer[o][i] * outputsHidden[i];
              }
            outO -= thresoldOutputLayer[o]; // Linear Activation

            // Денормализация для графики (жёлтая линия — прогноз на 1 бар)
            if(o == 0) yValues[iSample] = outO * range + localMin;

            targetCum += etalons[iSample + p + o];
            double targetNorm = (targetCum - localMin) / range;
            double errOutput = outO - targetNorm;
            // Чистая нормализованная ошибка (честная, без клиппинга — для диагностики/порогов)
            sumSqErr += errOutput * errOutput;

            // Gradient clipping: ограничиваем ошибку, от которой строится градиент, чтобы
            // редкий выброс (гэп/спайк) не мог разово раскрутить веса в NaN через momentum.
            if(errOutput > gradClipLimit) errOutput = gradClipLimit;
            else if(errOutput < -gradClipLimit) errOutput = -gradClipLimit;
            errOutClipped[o] = errOutput;
           }
         errGlobal += sumSqErr / countOutputBars;

         // Ошибка скрытого нейрона — сумма вкладов всех выходов.
         // Считается ДО обновления весов выходного слоя (как в v7.8.8).
         double errHiden[countHiddenNeuron];
         for(int i=0; i<countHiddenNeuron; i++)
           {
            double e = 0;
            for(int o=0; o<countOutputBars; o++)
              {
               e += errOutClipped[o] * weightsOutputLayer[o][i];
              }
            errHiden[i] = e * gradScale;
           }

         for(int o=0; o<countOutputBars; o++)
           {
            for(int i=0; i<countHiddenNeuron; i++)
              {
               double grad = errOutClipped[o] * gradScale * outputsHidden[i];
               vWeightsOutputLayer[o][i] = momentum * vWeightsOutputLayer[o][i] - a * grad;
               weightsOutputLayer[o][i] += vWeightsOutputLayer[o][i];
              }

            double gradThresoldOut = -errOutClipped[o] * gradScale;
            vThresoldOutputLayer[o] = momentum * vThresoldOutputLayer[o] - a * gradThresoldOut;
            thresoldOutputLayer[o] += vThresoldOutputLayer[o];
           }

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
                 double gradCandle = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * cacheCandleInputs[iSample][c];
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
                  double gradCorr1 = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * cacheCorr1Inputs[iSample][c];
                  vWeightsHidden[i][p+10+CANDLES_N*3+c] = momentum * vWeightsHidden[i][p+10+CANDLES_N*3+c] - a * gradCorr1;
                  weightsHidden[i][p+10+CANDLES_N*3+c] += vWeightsHidden[i][p+10+CANDLES_N*3+c];

                  double gradCorr2 = errHiden[i] * LeakyReLUDerivative(weightedSums[i]) * cacheCorr2Inputs[iSample][c];
                  vWeightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] = momentum * vWeightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] - a * gradCorr2;
                  weightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] += vWeightsHidden[i][p+10+CANDLES_N*3+CORR_N+c];
                 }
               
               double gradThresoldHid = -errHiden[i] * LeakyReLUDerivative(weightedSums[i]);
              vThresoldsHidden[i] = momentum * vThresoldsHidden[i] - a * gradThresoldHid;
              thresoldsHidden[i] += vThresoldsHidden[i];
           }
        }
      errGlobal = (errGlobal / trainSamples) * 100;
     }

   Print("Итерация " + countTeaches + " === Ошибко " + errGlobal + " Пох за бар " + epochsTrainedThisBar);
  }

//+------------------------------------------------------------------+
//| Predict with Dynamic Window Bounds                               |
//+------------------------------------------------------------------+
void Predict()
  {
   // v7.9: авторегрессии больше нет. Один проход сети: выход o — суммарный ход
   // за o+1 будущих баров. Обратно в по-барные дельты раскладываем разностями
   // соседних кумулятивных прогнозов, чтобы Trade() и отрисовка работали
   // в прежних терминах (predictValues[i] = дельта бара i).
   double localMin = 999999;
   double localMax = -999999;
   for(int j=0; j<p; j++)
     {
      double v = etalons[trainSize - p + j];
      if(v < localMin) localMin = v;
      if(v > localMax) localMax = v;
     }
   double range = localMax - localMin;
   if(range < 100 * _Point) range = 100 * _Point;

   double rsi = CalcRSI(etalons, trainSize - 1, 14);
   double rsiInput = (rsi - 50.0) / 100.0;
   double rsi2 = CalcRSI(etalons, trainSize - 1, 28);
   double rsiInput2 = (rsi2 - 50.0) / 100.0;

   double atrInput = 0;
   double atrInput2 = 0;
   if(useAtrAsInput)
     {
      double atr = iATR(NULL, 0, 14, 1);
      atrInput = 2.0 * MathArctan(atr / range) / 3.1415926535;
      double atr2 = iATR(NULL, 0, 100, 1);
      atrInput2 = 2.0 * MathArctan(atr2 / range) / 3.1415926535;
     }

   double maInput = 0;
   double maInput2 = 0;
   if(useMaAsInput)
     {
      double closePrice = iClose(NULL, 0, 1);
      double ma50 = iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE, 1);
      double ma200 = iMA(NULL, 0, 200, 0, MODE_SMA, PRICE_CLOSE, 1);
      maInput = 2.0 * MathArctan((closePrice - ma50) / range) / 3.1415926535;
      maInput2 = 2.0 * MathArctan((closePrice - ma200) / range) / 3.1415926535;
     }

   double candleInputs[CANDLES_N*3];
   for(int c=0; c<CANDLES_N*3; c++) candleInputs[c] = 0;
   if(useCandlesAsInput)
     {
      for(int c=0; c<CANDLES_N; c++)
        {
         int cShift = 1 + c;
         double cOpen = iOpen(NULL, 0, cShift);
         double cHigh = iHigh(NULL, 0, cShift);
         double cLow = iLow(NULL, 0, cShift);
         double cClose = iClose(NULL, 0, cShift);

         candleInputs[c*3] = 2.0 * MathArctan((cClose - cOpen) / range) / 3.1415926535;
         candleInputs[c*3+1] = 2.0 * MathArctan((cHigh - MathMax(cOpen, cClose)) / range) / 3.1415926535;
         candleInputs[c*3+2] = 2.0 * MathArctan((MathMin(cOpen, cClose) - cLow) / range) / 3.1415926535;
        }
     }

   double corr1Inputs[CORR_N];
   double corr2Inputs[CORR_N];
   for(int c=0; c<CORR_N; c++) { corr1Inputs[c] = 0; corr2Inputs[c] = 0; }
   if(useCorrelationAsInput)
     {
      for(int c=0; c<CORR_N; c++)
        {
         int cShift = 1 + c;
         double rsiC1 = iRSI(correlationSymbol1, 0, 14, PRICE_CLOSE, cShift);
         double rsiC2 = iRSI(correlationSymbol2, 0, 14, PRICE_CLOSE, cShift);
         corr1Inputs[c] = (rsiC1 - 50.0) / 100.0;
         corr2Inputs[c] = (rsiC2 - 50.0) / 100.0;
        }
     }

   double sinTime = 0;
   double cosTime = 0;
   double sinDay = 0;
   double cosDay = 0;
   if(useTimeAsInput || useDayOfWeekAsInput)
     {
      datetime barTime = iTime(NULL, 0, 1);
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

   double outH[countHiddenNeuron];
   for(int i=0; i<countHiddenNeuron; i++)
     {
      double wSum = 0;
      for(int j=0; j<p; j++)
        {
         double x = (etalons[trainSize - p + j] - localMin) / range;
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

   double cumPrev = 0;
   for(int o=0; o<countOutputBars; o++)
     {
      double outO = 0;
      for(int i=0; i<countHiddenNeuron; i++)
        {
         outO += weightsOutputLayer[o][i] * outH[i];
        }
      outO -= thresoldOutputLayer[o]; // Linear Activation

      // Денормализуем: cum — суммарный ход за o+1 баров в масштабе цены
      double cum = outO * range + localMin;
      predictValues[o] = cum - cumPrev; // дельта отдельного бара o
      cumPrev = cum;
     }

   // При ранней остановке торговое решение принимается снаружи, уже на лучших
   // весах — здесь Predict() вызывается много раз по ходу обучения и торговать
   // на промежуточных весах нельзя.
   if(useEarlyStopping) return;

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

   int total = countOutputBars > countPredictAnalysys ? countPredictAnalysys : countOutputBars;
   double predictedPrice = rawPrices[trainSize - 1]; // Начинаем от текущей цены закрытия

   // 1. Проматываем "эхо" (запаздывание)
   for(int j=0; j<predictShift; j++)
     {
      if(j < countOutputBars) predictedPrice += predictValues[j];
     }
     
   // 2. Фиксируем опорную точку (baseline) ПОСЛЕ запаздывания!
   double baselinePrice = predictedPrice; 

   for(int i=0; i<total; i++)
     {
      int idx = i + predictShift;
      if(idx >= countOutputBars) break;
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
   static int lastPeriodKey = -1;
   int periodKey = GetResetPeriodKey(Time[0]);
   if(lastPeriodKey == -1) lastPeriodKey = periodKey;

   if(periodKey != lastPeriodKey)
     {
      Print("==================================================================");
      if(useWeeklyReset) Print("Начало новой недели! Сбрасываем веса (Weekly Rolling Window)...");
      else               Print("Начало нового дня! Сбрасываем веса (Daily Rolling Window)...");
      InitWeights();
      lastResetTime = TimeCurrent();
      lastPeriodKey = periodKey;
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
      fastPassCounter++;

      // Новый бар — заново ищем лучшую точку обучения
      bestErrValidation = 999999;
      hasBestWeights = false;
      epochsSinceImprove = 0;
      skipTrainingThisBar = false;

      if(useEarlyStopping && validSamples > 0)
        {
         // Сдаём экзамен текущими весами на новых данных. Если сеть всё ещё
         // предсказывает лучше тупого прогноза — переучивать её незачем,
         // пропускаем обучение целиком и работаем на том, что есть.
         CalculateValidationError();

         bool stillGood = (errNaive > 0 && errValidation < errNaive * skipTrainRatio);

         if(allowSkipTrain && stillGood && fastPassCounter < skipTrainMaxPasses)
           {
            skipTrainingThisBar = true;
            bestErrValidation = errValidation;
            SaveBestWeights(); // текущие веса и есть лучшие, откатываться некуда
           }
         else
           {
            fastPassCounter = 0; // сеть перестала справляться — учим заново
           }
        }
      else if(allowSkipTrain)
        {
         CalculateError(); // старый путь: пропуск по ошибке на домашке
        }
     }

   if(ebobo && skipTrainingThisBar)
     {
      // Обучение не нужно. Просто строим прогноз на текущих весах и решаем, торговать ли.
      CalculateError(); // обновляем ошибку на домашке для отчёта
      Predict();

      if(showGraphics)
        {
         DrawLines();
         DrawLinesNN();
         DrawLinesPredict();
        }

      bool useful = (bestErrValidation < errNaive * beatNaiveRatio);
      if(!tradeOnlyIfBeatsNaive || useful)
        {
         Trade();
        }

      errValidation = bestErrValidation;
      ReportExam(true);

      ebobo = false;
     }

   if(ebobo)
     {
      // Один тик = один вызов Train() (то есть pass эпох). Обучение растянуто по
      // тикам бара специально: так видно, как жёлтая линия прогноза меняется в
      // реальном времени в визуальном тесте — по её форме подбирается порог ошибки.
      Train();
      epochsTrainedThisBar += pass; // Учитываем пройденные эпохи

      Predict(); // при выключенной ранней остановке сам вызовет Trade()

      if(showGraphics)
        {
         DrawLines();
         DrawLinesNN();
         DrawLinesPredict();
        }

      bool earlyStopActive = (useEarlyStopping && validSamples > 0);

      if(earlyStopActive)
        {
         // Сдаём экзамен после каждой порции обучения и следим, улучшается он
         // или уже начал портиться.
         CalculateValidationError();

         if(errValidation < bestErrValidation)
           {
            bestErrValidation = errValidation;
            SaveBestWeights();       // запоминаем момент наилучшего экзамена
            epochsSinceImprove = 0;
           }
         else
           {
            epochsSinceImprove += pass; // экзамен не улучшился — терпим
           }

         // Останавливаемся, если экзамен давно не улучшался (сеть пошла зубрить)
         // или если исчерпан лимит эпох.
         bool noProgress = (epochsSinceImprove >= earlyStopPatience);
         bool epochLimit = (epochsTrainedThisBar >= maxEpochsPerBar);

         if(noProgress || epochLimit)
           {
            RestoreBestWeights();  // откатываемся к лучшей точке
            CalculateError();      // пересчитываем ошибку на обучении для этих весов
            Predict();             // прогноз строим уже на лучших весах

            if(showGraphics)
              {
               // Именно DrawLines() чистит старые объекты NN и Predict. Без неё
               // ObjectCreate ниже упрётся в уже существующие линии (ошибка 4200).
               DrawLines();
               DrawLinesNN();
               DrawLinesPredict();
              }

            // Торговое решение: сеть должна быть полезнее тупого прогноза
            bool netIsUseful = (bestErrValidation < errNaive * beatNaiveRatio);
            if(!tradeOnlyIfBeatsNaive || netIsUseful)
              {
               Trade();
              }

            ebobo = false;
           }
        }
      else
        {
         // Лимит эпох исчерпан, а сеть так и не сошлась до торгового порога
         if(ebobo && epochsTrainedThisBar >= maxEpochsPerBar)
           {
            if(forceTradeOnEpochLimit)
              {
               Trade(); // Принудительно пытаемся открыть сделку по текущему прогнозу
              }
            ebobo = false; // Принудительно завершаем обучение на текущем баре
           }
        }

      // Обучение бара только что закончилось — записываем оценку за экзамен.
      // Ровно один раз на бар, а не на каждый тик.
      if(!ebobo && validSamples > 0)
        {
         if(!earlyStopActive) CalculateValidationError();
         else                 errValidation = bestErrValidation; // лучшая точка

         ReportExam(false);
        }
     }

   if(showGraphics) DrawInfoPanel();
  }

//+------------------------------------------------------------------+
//| Записать оценку за экзамен по итогам бара                         |
//+------------------------------------------------------------------+
void ReportExam(bool skipped)
  {
   if(validSamples <= 0) return;

   sumErrTrain += errGlobal;
   sumErrValidation += errValidation;
   sumErrNaive += errNaive;
   sumErrNaiveTrain += errNaiveTrain;
   sumDirCorrect += dirCorrectBar;
   sumDirTotal += dirTotalBar;
   sumDirHorCorrect += dirHorCorrectBar;
   sumDirHorTotal += dirHorTotalBar;
   sumDirSigCorrect += dirSigCorrectBar;
   sumDirSigTotal += dirSigTotalBar;
   sumDirLoudCorrect += dirLoudCorrectBar;
   sumDirLoudTotal += dirLoudTotalBar;
   validationBars++;

   double ratio = (errNaive > 0) ? errValidation / errNaive : 0;
   double dirPct = (dirTotalBar > 0) ? 100.0 * dirCorrectBar / dirTotalBar : 0;
   double dirHorPct = (dirHorTotalBar > 0) ? 100.0 * dirHorCorrectBar / dirHorTotalBar : 0;
   double dirSigPct = (dirSigTotalBar > 0) ? 100.0 * dirSigCorrectBar / dirSigTotalBar : 0;
   double dirLoudPct = (dirLoudTotalBar > 0) ? 100.0 * dirLoudCorrectBar / dirLoudTotalBar : 0;
   Print("ЭКЗАМЕН: обучение=", DoubleToString(errGlobal, 3),
         " | тупой.дом=", DoubleToString(errNaiveTrain, 3),
         " | спрятанные=", DoubleToString(errValidation, 3),
         " | тупой=", DoubleToString(errNaive, 3),
         " | сеть/тупой=", DoubleToString(ratio, 3),
         " | напр.1бар=", DoubleToString(dirPct, 1), "%",
         " | напр.", examDirBarsEff, "бар=", DoubleToString(dirHorPct, 1), "%",
         " | сигнал=", DoubleToString(dirSigPct, 1), "%",
         " | громкий=", DoubleToString(dirLoudPct, 1), "% (", dirLoudTotalBar, ")",
         " | эпох=", epochsTrainedThisBar,
         (skipped ? " | ОБУЧЕНИЕ ПРОПУЩЕНО" : ""));
  }

//+------------------------------------------------------------------+
//| Итоговый отчёт экзамена в конце теста                             |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(validationBars <= 0) return;

   double avgTrain = sumErrTrain / validationBars;
   double avgValid = sumErrValidation / validationBars;
   double avgNaive = sumErrNaive / validationBars;
   double avgNaiveTrain = sumErrNaiveTrain / validationBars;
   double ratio = (avgNaive > 0) ? avgValid / avgNaive : 0;
   double ratioTrain = (avgNaiveTrain > 0) ? avgTrain / avgNaiveTrain : 0;
   double dirPct = (sumDirTotal > 0) ? 100.0 * sumDirCorrect / sumDirTotal : 0;
   double dirHorPct = (sumDirHorTotal > 0) ? 100.0 * sumDirHorCorrect / sumDirHorTotal : 0;
   double dirSigPct = (sumDirSigTotal > 0) ? 100.0 * sumDirSigCorrect / sumDirSigTotal : 0;
   double dirLoudPct = (sumDirLoudTotal > 0) ? 100.0 * sumDirLoudCorrect / sumDirLoudTotal : 0;

   Print("==========================================================");
   Print("ИТОГ ЭКЗАМЕНА (среднее по ", validationBars, " барам)");
   Print("Спрятано от обучения примеров: ", validSamples, " из ", trainSize - p);
   Print("Ошибка на обучении (зазубренное): ", DoubleToString(avgTrain, 4));
   Print("Тупой прогноз на обучении:        ", DoubleToString(avgNaiveTrain, 4));
   Print("Ошибка на спрятанных (экзамен):   ", DoubleToString(avgValid, 4));
   Print("Ошибка тупого прогноза (планка):  ", DoubleToString(avgNaive, 4));
   Print("СЕТЬ / ТУПОЙ на обучении  = ", DoubleToString(ratioTrain, 4));
   Print("СЕТЬ / ТУПОЙ на экзамене  = ", DoubleToString(ratio, 4), "   (меньше 1.0 = сеть лучше тупого прогноза)");
   Print("НАПРАВЛЕНИЕ следующий бар   = ", DoubleToString(dirPct, 2), "%   (монетка = ~50%)");
   Print("НАПРАВЛЕНИЕ за ", examDirBarsEff, " баров (итоговый ход) = ", DoubleToString(dirHorPct, 2), "%   (монетка = ~50%)");
   Print("НАПРАВЛЕНИЕ сигнала Trade (шифт ", predictShift, ", анализ ", countPredictAnalysys, " баров) = ", DoubleToString(dirSigPct, 2), "%");
   Print("ГРОМКИЙ сигнал (перевес > ", orderThresold, " п.): точность = ", DoubleToString(dirLoudPct, 2),
         "%   (громких ", sumDirLoudTotal, " из ", sumDirSigTotal, ")");
   Print("==========================================================");
  }

//+------------------------------------------------------------------+
//| Информационная панель (только когда график реально виден)         |
//+------------------------------------------------------------------+
void DrawInfoPanel()
  {
   string status = (errGlobal < currentTradeErrThreshold) ? "[OK] READY (Trading)" : "[!] LEARNING";
   color txtColor = (color)ChartGetInteger(0, CHART_COLOR_FOREGROUND);
   ChartSetInteger(0, CHART_FOREGROUND, false);
   string lines[13];
   lines[0] = "--- Antigravity NN v7.9 (Multi-Out) ---";
   lines[1] = "Status: " + status;
   lines[2] = "Arch: Inp(" + IntegerToString(p+10+CANDLES_N*3+CORR_N*2) + ") -> Hid(" + IntegerToString(countHiddenNeuron) + ") -> Out(" + IntegerToString(countOutputBars) + " Linear, cum)";
   lines[3] = "Mem (trainSize): " + IntegerToString(trainSize) + " bars | RSI: Enabled (Dynamic)";
   lines[4] = "Global Error: " + DoubleToString(errGlobal, 5) + " / " + DoubleToString(currentTradeErrThreshold, 5) + (useDynamicError ? " (Dyn)" : "");
   lines[5] = "Passes Done: " + IntegerToString(countTeaches);
   lines[6] = "Pass (Epochs per tick): " + IntegerToString(pass);
   lines[7] = "Learning Rate (aStep): " + DoubleToString(aStep, 6);
   lines[8] = "--- Trading Logic ---";
   lines[9] = "Trade Horizon: " + IntegerToString(countPredictAnalysys) + " (Vis: " + IntegerToString(countOutputBars) + ") | Shift: " + IntegerToString(predictShift);
   
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
      double x1 = Time[trainSize - i + countOutputBars];
      double y1 = rawPrices[i] + offsetY;
      double x2 = Time[trainSize - i + countOutputBars-1];
      double y2 = rawPrices[i+1] + offsetY;
      int id = ObjectCreate("Ebat" + i, OBJ_TREND, 0, x1, y1, x2, y2);
      if(id == 0) Print(GetLastError());

      ObjectSetString(0, "Ebat" + i, OBJPROP_TEXT, id);
      ObjectSet("Ebat" + i, OBJPROP_COLOR, clrBlue);
      ObjectSet("Ebat" + i, OBJPROP_WIDTH, 5);
      ObjectSet("Ebat" + i, OBJPROP_RAY, 0);
     }
  }

//+------------------------------------------------------------------+
//| Жёлтая линия на хвосте: сэмплы после trainSamples в обучении не   |
//| участвуют (экзамен + последние countOutputBars-1 без полного      |
//| будущего), поэтому их yValues никто не обновляет и линия там      |
//| навсегда оставалась бы плоским нулём. Считаем для них 1-барный    |
//| прогноз (выход 0) чисто для отрисовки — на обучение, ошибку и     |
//| торговлю это не влияет.                                           |
//+------------------------------------------------------------------+
void FillTailYValues()
  {
   if(!showGraphics) return;

   PrecomputeSampleFeatures();

   for(int iSample=trainSamples; iSample<trainSize-p; iSample++)
     {
      double localMin = cacheLocalMin[iSample];
      double range = cacheRange[iSample];

      double outO = 0;
      for(int i=0; i<countHiddenNeuron; i++)
        {
         double wSum = 0;
         for(int j=0; j<p; j++)
           {
            double x = (etalons[j+iSample] - localMin) / range;
            wSum += weightsHidden[i][j] * x;
           }

         wSum += weightsHidden[i][p] * cacheRsiInput[iSample];
         wSum += weightsHidden[i][p+1] * cacheRsiInput2[iSample];
         wSum += weightsHidden[i][p+2] * cacheAtrInput[iSample];
         wSum += weightsHidden[i][p+3] * cacheAtrInput2[iSample];
         wSum += weightsHidden[i][p+4] * cacheMaInput[iSample];
         wSum += weightsHidden[i][p+5] * cacheMaInput2[iSample];
         for(int c=0; c<CANDLES_N*3; c++)
           {
            wSum += weightsHidden[i][p+6+c] * cacheCandleInputs[iSample][c];
           }
         wSum += weightsHidden[i][p+6+CANDLES_N*3] * cacheSinTime[iSample];
         wSum += weightsHidden[i][p+7+CANDLES_N*3] * cacheCosTime[iSample];
         wSum += weightsHidden[i][p+8+CANDLES_N*3] * cacheSinDay[iSample];
         wSum += weightsHidden[i][p+9+CANDLES_N*3] * cacheCosDay[iSample];
         for(int c=0; c<CORR_N; c++)
           {
            wSum += weightsHidden[i][p+10+CANDLES_N*3+c] * cacheCorr1Inputs[iSample][c];
            wSum += weightsHidden[i][p+10+CANDLES_N*3+CORR_N+c] * cacheCorr2Inputs[iSample][c];
           }

         wSum -= thresoldsHidden[i];
         outO += weightsOutputLayer[0][i] * LeakyReLU(wSum);
        }
      outO -= thresoldOutputLayer[0];

      yValues[iSample] = outO * range + localMin;
     }
  }

void DrawLinesNN()
  {
   FillTailYValues();

   int linesToDraw = trainSize-p;
   double currentY = rawPrices[p - 1];
   for(int i=0; i<linesToDraw; i++)
     {
      double x1 = Time[trainSize - i - p + countOutputBars + 1];
      double y1 = currentY + offsetY;
      currentY += yValues[i];
      double x2 = Time[trainSize - i - p + countOutputBars];
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
   for(int i=0; i<countOutputBars; i++)
     {
      double x1 = Time[countOutputBars - i + 1];
      double y1 = currentY;
      currentY += predictValues[i];
      double x2 = Time[countOutputBars - i];
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

// Ключ текущего периода сброса весов: меняется при смене дня (или недели).
// Для недели: 1970-01-01 — четверг, поэтому сдвиг на 4 дня переносит границу на понедельник 00:00.
int GetResetPeriodKey(datetime barTime)
  {
   if(useWeeklyReset)
     {
      int days = (int)(barTime / 86400);
      return (days - 4) / 7;
     }
   return TimeDay(barTime);
  }

// Абсолютный номер периода для сида (useDaySeed). В отличие от GetResetPeriodKey,
// дневной ключ здесь сквозной (номер дня с 1970), а не число месяца — иначе
// 15 апреля и 15 мая получали бы одинаковые стартовые веса.
int SeedPeriodKey()
  {
   int days = (int)(TimeCurrent() / 86400);
   if(useWeeklyReset) return (days - 4) / 7;
   return days;
  }
//+------------------------------------------------------------------+
