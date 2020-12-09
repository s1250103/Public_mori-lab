
function sT(){
  target = document.getElementById("id_start");
  const sTime = new Date();
  target.innerHTML = sTime.toLocaleString();
}

function fT(){
  target_start = document.getElementById("id_start");
  target_finish = document.getElementById("id_finish");
  target_spend = document.getElementById("id_spend");
  const fTime = new Date();
  const sTime = new Date(target_start.innerHTML);

  target_finish.innerHTML = fTime.toLocaleString();
  spend = calcSpend(sTime, fTime);
  hour = spend[0] + ' 時間 ';
  minute = spend[1] + ' 分 ';
  second = spend[2] + ' 秒';

  target_spend.innerHTML = hour +  minute +  second;
  save(sTime, fTime, hour, minute, second);
}

function calcSpend(sTime, fTime){
  mSecond = fTime.getTime() - sTime.getTime();
  second = mSecond / 1000;


  minute = second / 60;
  second = second % 60;

  hour = minute / 60;
  minute = minute % 60;

  second = parseInt(second);
  minute = parseInt(minute);
  hour = parseInt(hour);

  return [hour, minute, second];
}

function save(sTime, fTime, hour, minute, second){

  // 元のオブジェクト
  // const data = [
  //   "startTime" : sTime.toLocaleString(),
  //   "finishTime" : fTime.toLocaleString(),
  //   "spendTime" : hour+minute+second
  // ];

  const data = ["[startTime]: ", sTime.toLocaleString(),
                "[finishTime]: ", fTime.toLocaleString(),
                "[spendTime]: ", hour+minute+second];
  // // JSONに変換
  // const json_data = JSON.stringify(data);
  // console.log(json_data); // 確認用

  let blob = new Blob(data,{type:"text/plan"});

  let link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = sTime.toLocaleString() + '.txt';
  link.click();


}
