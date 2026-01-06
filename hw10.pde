float amplitude = 150; // 振幅 (彈簧拉伸的最大距離)
float angle = 0;       // 目前的角度
float angleVel = 0.05; // 角速度 (控制運動快慢)

void setup() {
  size(600, 400);
}

void draw() {
  background(220); // 淺灰色背景
  
  // 將座標原點移到左側中心基準點
  translate(width/4, height/2);
  
  // 1. 計算目前的位移 x
  // 使用 sin 或 cos 產生簡諧運動
  float x = amplitude * sin(angle);
  
  // 2. 繪製固定牆壁 (左側的直線)
  stroke(255);
  strokeWeight(4);
  line(0, -50, 0, 50);
  
  // 3. 繪製彈簧 (用鋸齒線或簡單直線模擬)
  stroke(255);
  strokeWeight(2);
  noFill();
  beginShape();
  int numPoints = 40; // 彈簧的圈數
  for (int i = 0; i <= numPoints; i++) {
    float px = map(i, 0, numPoints, 0, x + 200); // 彈簧終點跟隨方塊
    float py = (i % 2 == 0) ? -10 : 10;          // 上下抖動產生鋸齒感
    if (i == 0 || i == numPoints) py = 0;        // 頭尾平齊
    vertex(px, py);
  }
  endShape();
  
  // 4. 繪製藍色方塊 (物體)
  fill(0, 0, 255); // 藍色
  noStroke();
  rectMode(CENTER);
  rect(x + 200, 0, 40, 40); // 200 是初始位置偏移量
  
  // 5. 更新角度，讓它動起來
  angle += angleVel;
}
