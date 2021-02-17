import cv2
import glob
from pathlib import Path
import xmltodict, json
import matplotlib.pyplot as plt
import os 


def parse_data(root):
    fnames_mgp = glob.glob(root + '/*.mpg')
    for fname in fnames_mgp:
        fname_gt = fname[:-4] + '.xml'
        folder_name = os.path.basename(fname)        
        #names = glob.glob(root + '/*.mpg')
        #for f in fnames:
        fname = Path(fname)#Path(root) / 'Browse1.mpg'
        fname_gt = Path(fname_gt)#Path(root) / 'br1gt.xml'
        cap = cv2.VideoCapture(fname.__str__())
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames_path = Path(root) / folder_name[:-4]
        frames_path.mkdir(exist_ok=True)
    
        i = 0
        while(True):
            if i % 100 == 0:
                print("processing {}/{}".format(i, n_frames))
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            f = frames_path / (str(i).zfill(6) + '.png')
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(f.__str__(), frame)
            i = i + 1
            
        cap.release()
        cv2.destroyAllWindows()

        with open(fname_gt.__str__()) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())

        xml_file.close()    
        data_dict = data_dict['dataset']['frame']
        gt = dict()
        for i, v in enumerate(data_dict):
            if v['objectlist'] is not None:
                frame_id = int(v['@number'])
                obj = v['objectlist']['object']
                if isinstance(obj, list):
                    bbox = [] 
                    for d in obj:
                        '''rotate 90 deg'''
                        h = int(d['box']['@w'])
                        w = int(d['box']['@h'])
                        yc = int(d['box']['@xc'])
                        xc = h_frame-int(d['box']['@yc'])
                        id = int(d['@id'])
                        orientation = int(d['orientation'])
                        appearance = d['appearance']
                        bbox.append([xc-w/2, yc-h/2, xc+w/2, yc+h/2, id, orientation, appearance])
                    img_name = str(frame_id).zfill(6) + '.png'
                    gt[str(frame_id)] = {'bbox': bbox, 'img_name': img_name}
                else:
                    bbox = []
                    w = int(obj['box']['@w'])
                    h = int(obj['box']['@h'])
                    yc = int(obj['box']['@xc'])
                    xc = h_frame-int(obj['box']['@yc'])
                    id = int(obj['@id'])
                    orientation = int(obj['orientation'])
                    appearance = obj['appearance']
                    bbox.append([xc-w/2, yc-h/2, xc+w/2, yc+h/2, id, orientation, appearance])
                    img_name = str(frame_id).zfill(6) + '.png'
                    gt[str(frame_id)] = {'bbox': bbox, 'img_name': img_name}
        
        #img = cv2.imread((frames_path / gt['50']['img_name']).__str__())
        #bbox = gt['50']['bbox'][0]
        #cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
        #bbox = gt['50']['bbox'][1]
        #cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
        #plt.imshow(img)
        #plt.show()
        with open(frames_path.__str__()+'_gt.json', "w") as write_file:
            json.dump(gt, write_file)

if __name__ == "__main__":
    root = r'/home/core4/data/CAVIAR'
    parse_data(root)
