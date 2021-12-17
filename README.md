# Image Reflection Removal based onKnowledge-distilling Content Disentanglement

**Yan-Tsung Peng, Kai-Han Cheng, I-Sheng Fang, Wen-Yi Peng, and Jr-Shian Wu** <br>
<br>
## **Abstract:** <br>

_When we shoot pictures through transparent media, such as glass, reflection can undesirably occur, obscuring the scene we intended to capture. Therefore, removing reflection is practical in image restoration. However, a reflective scene mixed with that behind the glass is challenging to be separated, considered significantly ill-posed. This letter addresses the single image reflection removal (SIRR) problem by proposing a knowledge-distilling-based content disentangling model that can effectively decompose the transmission and reflection layers. The experiments on benchmark SIRR datasets demonstrate that our method performs favorably against state-of-the-art SIRR methods._
  ![image](https://github.com/Sasebalballgit/Reflection_Removal/blob/main/Example/architecture_SPL.png)
  
  
## Highlights <br>
  ![image](https://github.com/Sasebalballgit/Reflection_Removal/blob/main/Example/test1.gif)
  ![image](https://github.com/Sasebalballgit/Reflection_Removal/blob/main/Example/test2.gif)
## Environment <br>
  
*  Platforms: Windows 10 / cuda8.0 <br>
*  python: 3.7.6 / pytorch: 1.5 <br>
  

## Testing <br>
*  Change the root path、save image path in main.py. <br>
*  Move your testing image to /Test. <br>
*  Download our pretrained model from https://reurl.cc/Q7OKKp amd move to /Weighted. <br>
*  Evaluate the model performance by `"python main.py"`. <br>    


  
