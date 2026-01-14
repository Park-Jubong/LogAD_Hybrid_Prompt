<h2>실행 방법</h2>

<ol>
  <li>
    <b>데이터셋 다운로드</b><br/>
    <code>dataset</code> 폴더에 BGL, HDFS, Thunderbird 데이터셋을 다운로드합니다.
    (<a href="https://github.com/logpai/loghub" target="_blank" rel="noopener noreferrer">LogHub</a>)
    <pre><code># 예시: dataset 폴더 구성 (직접 다운로드/압축해제 후 배치)
dataset/
  bgl/
  hdfs/
  tbird/   (또는 thunderbird/)</code></pre>
  </li>

  <li>
    <b>LogBERT 클론</b><br/>
    메인 디렉토리에 LogBERT를 클론합니다.
    <pre><code>git clone https://github.com/HelenGuohx/logbert.git</code></pre>
  </li>

  <li>
    <b>main.sh 실행</b><br/>
    <code>main.sh</code>를 실행합니다.
    <ul>
      <li><code>preprocess</code>: 데이터셋 전처리 과정</li>
      <li><code>inference</code>: 로그 이상탐지 수행 과정</li>
      <li><code>confusion_matrix</code>: 이상탐지 결과(Confusion Matrix) 산출 과정</li>
    </ul>

    <p>
      <b>inference 수행 시 주의사항</b><br/>
      코드 내 OpenAI API Key를 입력해야 하며, inference 수행 시 prompt가 생성된 뒤 추론을 시작합니다.
    </p>

    <p>
      <b>전처리/추론 관련 옵션</b><br/>
      전처리 과정에서 <code>num_samples</code>(또는 <code>datasample</code>) 개수를 설정할 수 있으며,
      <code>compare_nv</code>, <code>cot_prompt</code>, <code>compare_sim</code>, <code>seq_info</code>는
      각각 복합 프롬프트의 구성 요소를 의미합니다.
    </p>

    <p>
      <b>Batch 추론(batch_llm) 옵션</b><br/>
      <code>batch_llm</code> args를 추가하면 OpenAI Batch를 통해 추론을 수행합니다.
      이 경우 추론 결과를 OpenAI에서 별도로 다운로드한 뒤 <code>output</code> 폴더로 옮기고,
      이후 <code>confusion_matrix</code>를 실행해야 합니다.
    </p>
  </li>
</ol>

<hr/>

<h3>예시 실행 방법</h3>
<p>예) 데이터셋에 대해 실행하는 방법</p>

<pre><code># 1) 전처리 (예: HDFS)
python main.py --mode preprocess --dataset hdfs

# 2) 추론 (예: BGL)
python main.py --mode inference --dataset bgl --num_samples 100 --compare_nv --cot_prompt --compare_sim --seq_info

# 3) 결과 산출 (예: BGL)
python main.py --mode confusion_matrix --dataset bgl
</code></pre>
