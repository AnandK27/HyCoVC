from src.helpers.basics import *


class Analysis_mv_net(nn.Module):
    '''
    Compress motion
    '''
    def __init__(self, out_channel_mv):
        super(Analysis_mv_net, self).__init__()
        self.conv1 = nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (2 + out_channel_mv) / (4))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.conv5 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv5.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv5.bias.data, 0.01)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)
        self.conv6 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv6.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv6.bias.data, 0.01)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1)
        self.conv7 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv7.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv7.bias.data, 0.01)
        self.relu7 = nn.LeakyReLU(negative_slope=0.1)
        self.conv8 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv8.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv8.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        return self.conv8(x)
class Synthesis_mv_net(nn.Module):
    '''
    Compress motion
    '''
    def __init__(self, out_channel_mv):
        super(Synthesis_mv_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv3 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv4 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv5 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv5.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv5.bias.data, 0.01)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv6 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv6.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv6.bias.data, 0.01)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv7 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv7.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv7.bias.data, 0.01)
        self.relu7 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv8 = nn.Conv2d(out_channel_mv, 2, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv8.weight.data, (math.sqrt(2 * 1 * (out_channel_mv + 2) / (out_channel_mv + out_channel_mv))))
        torch.nn.init.constant_(self.deconv8.bias.data, 0.01)

        
    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.relu3(self.deconv3(x))
        x = self.relu4(self.deconv4(x))
        x = self.relu5(self.deconv5(x))
        x = self.relu6(self.deconv6(x))
        x = self.relu7(self.deconv7(x))
        return self.deconv8(x)



def build_model():
    input_image = torch.zeros([4, 2, 256, 256])
    analysis_mv_net = Analysis_mv_net()
    synthesis_mv_net = Synthesis_mv_net()
    feature = analysis_mv_net(input_image)
    recon_image = synthesis_mv_net(feature)
    print(input_image.size())
    print(feature.size())
    print(recon_image.size())
    



if __name__ == '__main__':
    build_model()
