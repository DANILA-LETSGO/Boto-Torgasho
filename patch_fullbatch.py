import re

with open('/root/Boto-Torgasho/Experts/v21.mq4', 'r') as f:
    code = f.read()

init_code = """      double grad_wOut[countClasses][countHiddenNeuron]; ArrayInitialize(grad_wOut, 0);
      double grad_tOut[countClasses]; ArrayInitialize(grad_tOut, 0);
      double grad_wHid[countHiddenNeuron][p+10]; ArrayInitialize(grad_wHid, 0);
      double grad_tHid[countHiddenNeuron]; ArrayInitialize(grad_tHid, 0);
      
      for(int iSample=0; iSample <= maxSample; iSample++)"""
code = code.replace('      for(int iSample=0; iSample <= maxSample; iSample++)', init_code)

new_accum = """
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
"""

after_loop_adam = """
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
"""

pattern = r"(\s+double a = aStep;\s+adam_t\+\+;.*?)\s+errGlobal = \(errGlobal / \(maxSample \+ 1\)\);"
match = re.search(pattern, code, re.DOTALL)
if match:
    old_full_match = match.group(0)
    code = code.replace(old_full_match, new_accum + "        }\n" + after_loop_adam)
    with open('/root/Boto-Torgasho/Experts/v21.mq4', 'w') as f:
        f.write(code)
    print("Full-Batch patch applied successfully!")
else:
    print("Error: Could not find target block.")
