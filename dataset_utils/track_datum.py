from collections import defaultdict

from dataset_utils.label_obj import BaseObject


class TrackObjHandler:

    def __init__(self, root_path, seq_id):
        """

        Args:
            root_path:      path to seq folder
        """
        self.tracks = defaultdict(list)

        with open("{}/{}.txt".format(root_path, seq_id)) as fo:
            lines = fo.readlines()
            for line in lines:
                _temp = line.strip().split(' ')  # string containing <frame>, <id>, <y_min>, <x_min>, <y_max>, <x_max>
                frame_idx = int(_temp[0])
                x_c = float(_temp[2])
                y_c = float(_temp[3])
                w = float(_temp[4])
                h = float(_temp[5])
                _track = BaseObject(
                    y_min= y_c - h/2,
                    x_min= x_c- w/2,
                    y_max= y_c + h/2,
                    x_max= x_c + w/2,
                    track=int(_temp[1]),
                    category="None"
                )
                self.tracks[frame_idx].append(_track)

    @classmethod
    def deserialize(cls, line):
        """
        Alternate constructor to initialize from a string.
        Args:
            line:   s

        Returns:
            RotationObject instance
        """
