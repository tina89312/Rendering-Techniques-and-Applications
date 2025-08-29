# 成像技術與應用作業簡介

「成像技術與應用」課程中完成的作業，內容涵蓋影像加密、矩陣轉換、資料隱藏與整合性保護等。以下依作業編號簡介每個主題的核心目標與實作內容。

---

## Assignment 01 - Image Encryption by 2D EAT and RP

使用 2D EAT（Extended Affine Transform）搭配 Durstenfeld 的 Random Permutation 對影像進行像素座標與位元加密。加密後的影像會被打亂內容與位元結構，並實作相對應的解密程式來還原影像。支援灰階與彩色圖片。

**重點**：空間與位元層級的雙重加密設計。

---

## Assignment 02 - Image Encryption Using Enhanced Sine Chaotic Sequence

在第一份作業基礎上加入混沌系統（2D SLM）進行像素擾動與擴散，並根據影像屬性動態產生參數，提高安全性與亂度。密鑰參數也會寫入 `Secret-Key.txt` 中，解密時讀入使用。

**重點**：將混沌理論導入圖像加密流程，加強擴散效果。

---

## Assignment 03 - Rectangular Transformation Matrix Finder

針對任意輸入影像解析度 (M, N)，窮舉並找出合法的矩陣參數 (a, b, c, d)，這些參數能對影像做矩形轉換 (RT)，且符合指定的數論條件。程式會輸出每組合法參數及其對應的週期值。

**重點**：建立合法轉換矩陣組合並計算其週期。

---

## Assignment 04 - Determine Period from Coefficients

輸入特定的矩陣參數與影像尺寸，判斷此轉換矩陣是否為合法 RT 矩陣，若合法則計算影像在此轉換下回到原始狀態所需的週期數。

**重點**：驗證轉換合法性與週期性。

---

## Assignment 05 - Inverse Rectangular Transformation

根據 RT/IRT 演算法，實作加密與解密的完整流程。使用指定密鑰參數 (包含 a, b, c, d, G) 操作影像座標轉換，測量執行時間並記錄。

**重點**：實作影像座標轉換的正逆向操作流程。

---

## Assignment 06 - GWM Data Hiding

使用 General Weighted Modulus (GWM) 演算法將秘密訊息嵌入灰階與彩色影像中。透過不同進位制與加權表格（PA Table）設計嵌入訊號，並處理 pixel overflow 問題。

**重點**：多進制資料隱藏技術與嵌入正確性驗證。

---

## Assignment 07 - GWMRDH Algorithm (Reversible Data Hiding)

將 GWM 擴展為可逆的資料隱藏演算法（GWMRDH），實作三張影像共同嵌入與回復過程，並驗證 MSE、PSNR 等品質指標，確保嵌密與還原效果。

**重點**：支援資料完整回復的影像隱藏架構。

---

## Assignment 08 - Integrated Message Embedding and Encryption Algorithm

結合 Assignment 05 和 07 的概念，建立完整的偽裝加密與嵌密系統（FCUE），以及對應的解密與取密流程（BDIX）。涵蓋嵌密、通道組合、隨機排列、座標加密等流程。

**重點**：資料嵌密 + 加密整合的多階段防護架構。

---

