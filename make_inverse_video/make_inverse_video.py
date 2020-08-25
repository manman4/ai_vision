import cv2

def make_inverse_video(filename):
    name       = filename.split(".")[0]
    extentsion = filename.split(".")[1]
    cap = cv2.VideoCapture(filename)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output_filename = name + "2." + extentsion
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w,h))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            """
            1  :左右反転
            2  :上下反転
            -1 :上下左右反転
            """
            frame = cv2.flip(frame,2)

            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

make_inverse_video("walking.mp4")
