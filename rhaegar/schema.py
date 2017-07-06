class TumblrItem:
    def __init__(self, row):
        self.username = row[0]
        self.post_id = row[1]
        self.date = row[2]
        self.url = row[3]
        self.content = row[4]
        self.is_owner = row[5] == 'True'

    def to_array(self):
        return [self.username, self.post_id, self.date, self.url, self.content, self.is_owner]

class TwitterItem:
    def __init__(self, row):
        self.username = row[0]
        self.content = row[1]
        self.date = row[2]
        self.is_owner = row[3] == 'True'

    def to_array(self):
        return [self.username, self.content, self.date, self.is_owner]

