import cv2
import glob
from pathlib import Path
import xmltodict, json
import matplotlib.pyplot as plt
import os 


def parse_data(root):
    fnames_mgp = sorted(glob.glob(root + '/*.mpg'))
    for fname in fnames_mgp:                                               
        print(fname)
        fname_gt = fname[:-4] + '.xml'
        subset = os.path.basename(os.path.dirname(fname))
        folder_name = os.path.basename(fname)        
        fname = Path(fname)
        fname_gt = Path(fname_gt)
        cap = cv2.VideoCapture(fname.__str__())
        fps = int(cap.get(cv2.CAP_PROP_FPS))
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
            if subset == 'mall':
                pass
            elif subset == 'loby':                
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)                
            else:
                raise NotImplementedError

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
            #skip close frames
            if i % (fps // 2) != 0:
                continue
            if v['objectlist'] is not None:
                frame_id = int(v['@number'])
                obj = v['objectlist']['object']
                if isinstance(obj, list):
                    bbox = []
                    for d in obj:                        
                        #don't rotate 90 degs
                        if subset == 'mall':
                            w = int(d['box']['@w'])
                            h = int(d['box']['@h'])
                            yc = int(d['box']['@yc'])
                            xc = int(d['box']['@xc'])
                            x_min = float(max(xc-w/2, 0))
                            y_min = float(max(yc-h/2, 0))
                            x_max = float(min(xc+w/2, w_frame-1))
                            y_max = float(min(yc+h/2, h_frame-1))
                        elif subset == 'loby':
                            '''rotate 90 deg'''
                            h = int(d['box']['@w'])
                            w = int(d['box']['@h'])
                            yc = int(d['box']['@xc'])
                            xc = h_frame-int(d['box']['@yc'])
                            x_min = float(max(xc-w/2, 0))
                            y_min = float(max(yc-h/2, 0))
                            x_max = float(min(xc+w/2, h_frame-1))
                            y_max = float(min(yc+h/2, w_frame-1))
                        else:
                            raise NotImplementedError

                        id = int(d['@id'])
                        orientation = int(d['orientation'])
                        appearance = d['appearance']                        
                        if x_min >= x_max or y_min >= y_max:
                            continue
                        bbox.append([x_min, y_min, x_max, y_max, id, orientation, appearance])
                    img_name = subset + '/' + folder_name[:-4] + '/' + str(frame_id).zfill(6) + '.png'
                    gt[img_name] = {'bbox': bbox, 'img_name': img_name}
                else:
                    bbox = []                    
                    if subset == 'mall':
                        h = int(d['box']['@h'])
                        w = int(d['box']['@w'])
                        xc = int(obj['box']['@xc'])
                        yc = int(obj['box']['@yc'])
                        x_min = float(max(xc-w/2, 0))
                        y_min = float(max(yc-h/2, 0))
                        x_max = float(min(xc+w/2, w_frame-1))
                        y_max = float(min(yc+h/2, h_frame-1))
                    elif subset == 'loby':
                        '''rotate 90 deg'''
                        h = int(d['box']['@w'])
                        w = int(d['box']['@h'])
                        yc = int(obj['box']['@xc'])
                        xc = h_frame-int(obj['box']['@yc'])
                        x_min = float(max(xc-w/2, 0))
                        y_min = float(max(yc-h/2, 0))
                        x_max = float(min(xc+w/2, h_frame-1))
                        y_max = float(min(yc+h/2, w_frame-1))
                    else:
                        raise NotImplementedError
                    id = int(obj['@id'])
                    orientation = int(obj['orientation'])
                    appearance = obj['appearance']                    
                    if x_min >= x_max or y_min >= y_max:
                        continue
                    bbox.append([x_min, y_min, x_max, y_max, id, orientation, appearance])
                    img_name = subset + '/' + folder_name[:-4] + '/' + str(frame_id).zfill(6) + '.png'
                    gt[img_name] = {'bbox': bbox, 'img_name': img_name}
                
        with open(frames_path.__str__()+'_gt.json', "w") as write_file:
            json.dump(gt, write_file)

if __name__ == "__main__":
    root = [r'/home/core4/data/CAVIAR/loby', r'/home/core4/data/CAVIAR/mall']
    for r in root:    
        parse_data(r)
