import argparse
import cv2
import onnxruntime
import numpy as np


class nighttime_dehaze:
    def __init__(self, modelpath):
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession(modelpath)
        self.input_name = self.onnx_session.get_inputs()[0].name
        _, _, self.input_height, self.input_width = self.onnx_session.get_inputs()[0].shape

    def detect(self, image):
        input_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dsize=(
            self.input_width, self.input_height))
        input_image = (input_image.astype(np.float32) / 255.0 - 0.5) / 0.5
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)

        result = self.onnx_session.run(None, {self.input_name: input_image})
        ###nighttime_dehaze_realnight_Nx3xHxW.onnx和nighttime_dehaze_realnight_1x3xHxW.onnx在run函数这里会出错
        # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
        output_image = np.squeeze(result[0])
        output_image = output_image.transpose(1, 2, 0)
        output_image = cv2.resize(output_image, (image.shape[1], image.shape[0]))
        output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        return output_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str,
                        default='images/511.png', help="image path")
    parser.add_argument('--modelpath', type=str,
                        default='weights/nighttime_dehaze_realnight_1x3x512x512.onnx', help="onnx path")
    args = parser.parse_args()

    mynet = nighttime_dehaze(args.modelpath)
    srcimg = cv2.imread(args.imgpath)

    dstimg = mynet.detect(srcimg)

    if srcimg.shape[0] > srcimg.shape[1]:
        boundimg = np.zeros((10, srcimg.shape[1], 3), dtype=srcimg.dtype)+255  ###中间分开原图和结果
        combined_img = np.vstack([srcimg, boundimg, dstimg])
    else:
        boundimg = np.zeros((srcimg.shape[0], 10, 3), dtype=srcimg.dtype)+255
        combined_img = np.hstack([srcimg, boundimg, dstimg])
    winName = 'Deep Learning in onnxruntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, combined_img)  ###原图和结果图也可以分开窗口显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()
